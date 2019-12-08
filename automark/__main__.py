import argparse
from automark.preprocess import preprocess
from automark.train import train
from automark.test import generate

def main():
    ap = argparse.ArgumentParser("AutoMark")

    ap.add_argument("mode", choices=["preprocess", "train", "generate"], help="preprocess data, train a model, or generate markings")

    ap.add_argument("config_path", type=str, help="path to YAML config file")
    ap.add_argument("input_src", type=str, help="path to input src")
    ap.add_argument("input_mt", type=str, help="path to input MT hyps")
    ap.add_argument("output_path", type=str, help="path to output file")

    args = ap.parse_args()
    print(args)
    if args.mode == 'preprocess':
        preprocess(args.config_path)
    elif args.mode == 'train':
        train(args.config_path)
    elif args.mode == "generate":
        generate(args.config_path, args.input_src, args.input_mt,
                 args.output_path)


if __name__ == "__main__":
    main()
