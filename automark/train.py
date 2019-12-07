import time
import queue
import shutil
import os

import torch
from torch.nn import functional as F

from automark.mark import AutoMark, build_model
from automark.dataset import make_dataset, make_data_iter
from automark.helpers import *
from automark.loss import XentLoss
from automark.builders import build_optimizer, build_scheduler, build_gradient_clipper
from automark.batch import Batch
from automark.test import test
from automark.validate import validate_on_data


class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model, config):
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        """
        train_config = config["train"]

        # files for logging and storing
        self.model_dir = make_model_dir(
            train_config["model_dir"], overwrite=train_config.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = train_config.get("logging_freq", 10)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        self.eval_metric = "loss"

        # model
        self.model = model
        self.pad_index = self.model.pad_index
        self.bos_index = self.model.bos_index
        self._log_parameters_list()

        # objective
        # print("ignore index {}".format(self.pad_index))
        self.loss = XentLoss()  # ignore_id=self.pad_index)
        self.normalization = train_config.get("normalization", "tokens")
        if self.normalization not in ["batch", "tokens"]:
            raise ConfigurationError(
                "Invalid normalization. " "Valid options: 'batch', 'tokens'."
            )

        self.weighting = train_config["weighting"]

        # optimization
        self.learning_rate_min = train_config.get("learning_rate_min", 1.0e-8)

        self.clip_grad_fun = build_gradient_clipper(config=train_config)

        opt_params = [{'params': self.model.marking_head.parameters(), 'lr': train_config['lr']}, {'params': self.model.bert.parameters(), 'lr': train_config.get('bert_lr', train_config['lr'])}]
        self.optimizer = build_optimizer(config=train_config,
                                         parameters=opt_params)

        # validation & early stopping
        self.validation_freq = train_config.get("validation_freq", 1000)
        self.log_valid_sents = train_config.get("print_valid_sents", [0, 1, 2])
        self.ckpt_queue = queue.Queue(maxsize=train_config.get("keep_last_ckpts", 5))

        self.early_stopping_metric = train_config.get(
            "early_stopping_metric", "eval_metric"
        )

        # if we schedule after BLEU/chrf, we want to maximize it, else minimize
        # early_stopping_metric decides on how to find the early stopping point:
        # ckpts are written when there's a new high/low score for this metric
        if self.early_stopping_metric in ["ppl", "loss"]:
            self.minimize_metric = True
        else:
            raise ConfigurationError(
                "Invalid setting for 'early_stopping_metric', "
                "valid options: 'loss', 'ppl', 'eval_metric'."
            )

        # learning rate scheduling
        self.scheduler, self.scheduler_step_at = build_scheduler(
            config=train_config,
            scheduler_mode="min" if self.minimize_metric else "max",
            optimizer=self.optimizer,
            hidden_size=model.bert.config.hidden_size,
        )

        # data & batch handling
        self.shuffle = train_config.get("shuffle", True)
        self.epochs = train_config["epochs"]
        self.batch_size = train_config["batch_size"]
        self.eval_batch_size = train_config.get("eval_batch_size", self.batch_size)

        self.batch_multiplier = train_config.get("batch_multiplier", 1)

        # CPU / GPU
        self.use_cuda = train_config["cuda"]
        if self.use_cuda:
            self.model.cuda()
            self.loss.cuda()

        # initialize training statistics
        self.steps = 0
        # stop training if this flag is True by reaching learning rate minimum
        self.stop = False
        self.total_tokens = 0
        self.best_ckpt_iteration = 0
        # initial values for best scores
        self.best_ckpt_score = np.inf if self.minimize_metric else -np.inf
        # comparison function for scores
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # model parameters
        if "load_model" in train_config.keys():
            model_load_path = train_config["load_model"]
            self.logger.info("Loading model from %s", model_load_path)
            self.init_from_checkpoint(model_load_path)

    def _save_checkpoint(self) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.

        The training state contains the total number of training steps,
        the total number of training tokens,
        the best checkpoint score and iteration so far,
        and optimizer and scheduler states.

        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "best_ckpt_score": self.best_ckpt_score,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()  # delete oldest ckpt
            try:
                os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        # create/modify symbolic link for best checkpoint
        symlink_update(
            "{}.ckpt".format(self.steps), "{}/best.ckpt".format(self.model_dir)
        )

    def init_from_checkpoint(self, path: str) -> None:
        """
        Initialize the trainer from a given checkpoint file.

        This checkpoint file contains not only model parameters, but also
        scheduler and optimizer states, see `self._save_checkpoint`.

        :param path: path to checkpoint
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        # restore model and optimizer parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])

        if (
            model_checkpoint["scheduler_state"] is not None
            and self.scheduler is not None
        ):
            self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])

        # restore counts
        self.steps = model_checkpoint["steps"]
        self.total_tokens = model_checkpoint["total_tokens"]
        self.best_ckpt_score = model_checkpoint["best_ckpt_score"]
        self.best_ckpt_iteration = model_checkpoint["best_ckpt_iteration"]

        # move parameters to cuda
        if self.use_cuda:
            self.model.cuda()

    def train_and_validate(self, train_data: Dataset, valid_data: Dataset) -> None:
        """
        Train the model and validate it from time to time on the validation set.

        :param train_data: training data
        :param valid_data: validation data
        """
        train_iter = make_data_iter(
            train_data, batch_size=self.batch_size, train=True, shuffle=self.shuffle
        )
        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no + 1)

            if self.scheduler is not None and self.scheduler_step_at == "epoch":
                self.scheduler.step(epoch=epoch_no)

            self.model.train()

            start = time.time()
            total_valid_duration = 0
            processed_tokens = self.total_tokens
            count = self.batch_multiplier - 1
            epoch_loss = 0

            for batch in iter(train_iter):
                # reactivate training
                self.model.train()
                # create a Batch object from torchtext batch

                # only update every batch_multiplier batches
                # see https://medium.com/@davidlmorton/
                # increasing-mini-batch-size-without-increasing-
                # memory-6794e10db672
                update = count == 0
                # print(count, update, self.steps)
                batch_loss, ones, acc = self._train_batch(batch, update=update)
                self.tb_writer.add_scalar(
                    "train/train_batch_loss", batch_loss, self.steps
                )
                self.tb_writer.add_scalar("train/train_batch_ones", ones, self.steps)
                self.tb_writer.add_scalar("train/train_batch_acc", acc, self.steps)
                count = self.batch_multiplier if update else count
                count -= 1
                epoch_loss += batch_loss.detach().cpu().numpy()

                if (
                    self.scheduler is not None
                    and self.scheduler_step_at == "step"
                    and update
                ):
                    self.scheduler.step()

                # log learning progress
                if self.steps % self.logging_freq == 0 and update:
                    elapsed = time.time() - start - total_valid_duration
                    elapsed_tokens = self.total_tokens - processed_tokens
                    self.logger.info(
                        "Epoch %3d Step: %8d Batch Loss: %12.6f Ones: %.2f "
                        "Accuracy: %.2f Tokens per Sec: %8.0f, Lr: %.6f",
                        epoch_no + 1,
                        self.steps,
                        batch_loss,
                        ones,
                        acc,
                        elapsed_tokens / elapsed,
                        self.optimizer.param_groups[0]["lr"],
                    )
                    start = time.time()
                    total_valid_duration = 0

                # validate on the entire dev set
                if self.steps % self.validation_freq == 0 and update:
                    print("Validating")
                    valid_start_time = time.time()

                    valid_score, valid_loss, valid_ones, f1 = validate_on_data(
                        batch_size=self.eval_batch_size,
                        data=valid_data,
                        eval_metric=self.eval_metric,
                        model=self.model,
                        use_cuda=self.use_cuda,
                        loss_function=self.loss,
                    )

                    self.tb_writer.add_scalar(
                        "valid/valid_loss", valid_loss, self.steps
                    )
                    self.tb_writer.add_scalar(
                        "valid/valid_score", valid_score, self.steps
                    )
                    self.tb_writer.add_scalar(
                        "valid/valid_ones", valid_ones, self.steps
                    )

                    self.tb_writer.add_scalar("valid/valid_f1", f1, self.steps)

                    if self.early_stopping_metric == "loss":
                        ckpt_score = valid_loss
                    else:
                        ckpt_score = valid_score

                    new_best = False
                    if self.is_best(ckpt_score):
                        self.best_ckpt_score = ckpt_score
                        self.best_ckpt_iteration = self.steps
                        self.logger.info(
                            "Hooray! New best validation result [%s]!",
                            self.early_stopping_metric,
                        )
                        if self.ckpt_queue.maxsize > 0:
                            self.logger.info("Saving new checkpoint.")
                            new_best = True
                            self._save_checkpoint()

                    if (
                        self.scheduler is not None
                        and self.scheduler_step_at == "validation"
                    ):
                        self.scheduler.step(ckpt_score)

                    # append to validation report
                    self._add_report(
                        valid_score=valid_score,
                        valid_loss=valid_loss,
                        valid_ones=valid_ones,
                        valid_f1=f1,
                        eval_metric=self.eval_metric,
                        new_best=new_best,
                    )

                    valid_duration = time.time() - valid_start_time
                    total_valid_duration += valid_duration
                    self.logger.info(
                        "Validation result at epoch %3d, step %8d: %s: %6.2f, "
                        "loss: %8.4f, ones: %8.4f, f1: %8.4f, duration: %.4fs",
                        epoch_no + 1,
                        self.steps,
                        self.eval_metric,
                        valid_score,
                        valid_loss,
                        valid_ones,
                        f1,
                        valid_duration,
                    )

                    # store validation set outputs

                if self.stop:
                    break
            if self.stop:
                self.logger.info(
                    "Training ended since minimum lr %f was reached.",
                    self.learning_rate_min,
                )
                break

            self.logger.info(
                "Epoch %3d: total training loss %.2f", epoch_no + 1, epoch_loss
            )
        else:
            self.logger.info("Training ended after %3d epochs.", epoch_no + 1)
        self.logger.info(
            "Best validation result at step %8d: %6.2f %s.",
            self.best_ckpt_iteration,
            self.best_ckpt_score,
            self.early_stopping_metric,
        )

        self.tb_writer.close()  # close Tensorboard writer

    def _train_batch(self, batch: Batch, update: bool = True) -> (Tensor, float):
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return: loss for batch (sum)
        """
        batch_loss, ones, acc, _ = self.model.get_loss_for_batch(
            batch, self.loss, self.weighting
        )

        # normalize batch loss
        if self.normalization == "batch":
            normalizer = batch.src_trg[0].shape[0]
        elif self.normalization == "tokens":
            # print("trg len {}".format(batch.trg_len))
            normalizer = torch.sum(batch.trg_len)
        else:
            raise NotImplementedError("Only normalize by 'batch' or 'tokens'")
        # print("Batch loss {}".format(batch_loss))
        norm_batch_loss = batch_loss / normalizer
        # print("Normalized loss {} ({})".format(norm_batch_loss, normalizer))
        # print("Proportion of predicted 1s: {:.2f}".format(ones))
        # division needed since loss.backward sums the gradients until updated
        norm_batch_multiply = norm_batch_loss / self.batch_multiplier

        # compute gradients
        norm_batch_multiply.backward()

        if self.clip_grad_fun is not None:
            # clip gradients (in-place)
            self.clip_grad_fun(params=self.model.parameters())

        if update:
            # make gradient step
            self.optimizer.step()
            self.optimizer.zero_grad()

            # increment step counter
            self.steps += 1

        # increment token counter
        self.total_tokens += torch.sum(batch.trg_len).item()

        return norm_batch_loss, ones, acc

    def _add_report(
        self,
        valid_score: float,
        valid_loss: float,
        valid_ones: float,
        valid_f1: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        """
        Append a one-line report to validation logging file.

        :param valid_score: validation evaluation score [eval_metric]
        :param valid_ppl: validation perplexity
        :param valid_loss: validation loss (sum over whole validation set)
        :param eval_metric: evaluation metric, e.g. "bleu"
        :param new_best: whether this is a new best model
        """
        current_lr = -1
        # ignores other param groups for now
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, "a") as opened_file:
            opened_file.write(
                "Steps: {}\tLoss: {:.5f}\t{}: {:.5f}\tOnes: {:.5f}\tF1: {:.5f}"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    valid_loss,
                    eval_metric,
                    valid_score,
                    valid_ones,
                    valid_f1,
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params


"""
    def _log_examples(self, sources: List[str], hypotheses: List[str],
                      references: List[str],
                      sources_raw: List[List[str]] = None,
                      hypotheses_raw: List[List[str]] = None,
                      references_raw: List[List[str]] = None) -> None:
        
        Log a the first `self.log_valid_sents` sentences from given examples.

        :param sources: decoded sources (list of strings)
        :param hypotheses: decoded hypotheses (list of strings)
        :param references: decoded references (list of strings)
        :param sources_raw: raw sources (list of list of tokens)
        :param hypotheses_raw: raw hypotheses (list of list of tokens)
        :param references_raw: raw references (list of list of tokens)
        
        for p in self.log_valid_sents:

            if p >= len(sources):
                continue

            self.logger.info("Example #%d", p)

            if sources_raw is not None:
                self.logger.debug("\tRaw source:     %s", sources_raw[p])
            if references_raw is not None:
                self.logger.debug("\tRaw reference:  %s", references_raw[p])
            if hypotheses_raw is not None:
                self.logger.debug("\tRaw hypothesis: %s", hypotheses_raw[p])

            self.logger.info("\tSource:     %s", sources[p])
            self.logger.info("\tReference:  %s", references[p])
            self.logger.info("\tHypothesis: %s", hypotheses[p])

    def _store_outputs(self, hypotheses: List[str]) -> None:
        
        Write current validation outputs to file in `self.model_dir.`

        :param hypotheses: list of strings
        
        current_valid_output_file = "{}/{}.hyps".format(self.model_dir,
                                                        self.steps)
        with open(current_valid_output_file, 'w') as opened_file:
            for hyp in hypotheses:
                opened_file.write("{}\n".format(hyp))
"""


def train(cfg_file):
    """
    Main training function. After training, also test on test data if given.

    :param cfg_file: path to configuration yaml file
    """
    cfg = load_config(cfg_file)

    # set the random seed
    set_seed(seed=cfg["train"].get("seed", 42))

    # load the data
    train_data, dev_data, test_data = make_dataset(cfg)

    # build an encoder-decoder model
    model = build_model(cfg)

    # for training management, e.g. early stopping and model selection
    trainer = TrainManager(model=model, config=cfg)

    # store copy of original training config in model dir
    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")

    # log all entries of config
    log_cfg(cfg, trainer.logger)

    trainer.logger.info(str(model))

    # train the model
    trainer.train_and_validate(train_data=train_data, valid_data=dev_data)

    # predict with the best model on validation and test
    # (if test data is available)
    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "{:08d}.hyps".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    test(cfg_file, ckpt=ckpt, output_path=output_path, logger=trainer.logger)
