import torch
from automark.helpers import load_config, load_checkpoint
from automark.dataset import make_generate_data, make_data_iter, batch_to
from automark.mark import build_model


def generate(cfg_file, input_src, input_mt, output_path=None):
    cfg = load_config(cfg_file)
    batch_size = cfg["generate"].get("batch_size", 1)
    cuda = cfg["generate"].get("cuda", False)

    # load the data (only inputs, no labels)
    data = make_generate_data(cfg, input_src, input_mt)

    # build an encoder-decoder model
    model = build_model(cfg)

    # load model from checkpoint
    ckpt = load_checkpoint(
        cfg["train"]["model_dir"]+"/best.ckpt", use_cuda=cuda)
    model.load_state_dict(ckpt["model_state"])

    data_iter = make_data_iter(data, batch_size, False, False)

    model.eval()

    with open(output_path, "w") as ofile:

        with torch.no_grad():
            for i, input_batch in enumerate(data_iter):
                if cuda:
                    input_batch = batch_to(input_batch, "cuda")

                predictions = model.predict(input_batch).argmax(-1).cpu().numpy()
                label_mask = input_batch.id_mask.cpu().numpy()
                assert predictions.shape == label_mask.shape

                for p, m in zip(predictions, label_mask):
                    valid_labels = [str(pi) for pi, mi in zip(p, m) if mi]
                    ofile.write("{}\n".format(" ".join(valid_labels)))


