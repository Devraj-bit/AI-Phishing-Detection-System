"""
Classify text as phishing or legitimate using the trained model.
Usage:
  python predict.py "Your text or URL here"
  python predict.py --file input.txt
"""
import pickle
import sys
from pathlib import Path

def load_model():
    out_dir = Path(__file__).resolve().parent
    path = out_dir / "model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            "model.pkl not found. Run train_model.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)

def predict(text: str) -> str:
    pipeline = load_model()
    pred = pipeline.predict([text])[0]
    return pred

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"text or URL\"")
        print("   or: python predict.py --file path/to/file.txt")
        sys.exit(1)

    if sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Usage: python predict.py --file path/to/file.txt")
            sys.exit(1)
        with open(sys.argv[2], "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = " ".join(sys.argv[1:])

    if not text.strip():
        print("No text to classify.")
        sys.exit(1)

    label = predict(text)
    print(f"Prediction: {label}")

if __name__ == "__main__":
    main()
