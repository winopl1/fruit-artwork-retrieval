import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("mps available:", torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        print("recommended device: mps")
    elif torch.cuda.is_available():
        print("recommended device: cuda")
    else:
        print("recommended device: cpu")


if __name__ == "__main__":
    main()
