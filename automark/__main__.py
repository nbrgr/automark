import argparse
from automark.preprocess import preprocess

def main():
    ap = argparse.ArgumentParser("AutoMark")

    ap.add_argument("mode", choices=["preprocess", "train", "generate"], help="preprocess data, train a model, or generate markings")

    ap.add_argument("config_path", type=str, help="path to YAML config file")

    args = ap.parse_args()
    print(args)
    if args.mode == 'preprocess':
        preprocess(args.config_path)

if __name__ == "__main__":
    main()
