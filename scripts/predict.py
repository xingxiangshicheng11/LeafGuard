import argparse


def main():
    parser = argparse.ArgumentParser(description="LeafGuard image prediction")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--classes", default="splits/class_to_idx.json")
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    args = parser.parse_args()
    raise NotImplementedError(
        "Prediction will be implemented after checkpoint serialization is finalized. "
        f"Arguments received: {args}"
    )


if __name__ == "__main__":
    main()
