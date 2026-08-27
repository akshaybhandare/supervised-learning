import os
import sys
import pickle
import warnings
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer


class NoteClassifier:
    def __init__(self, model_path: str = None):
        """Load trained neural network weights and pre-trained embedding model."""
        if model_path is None or not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")

        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.id2label = checkpoint["id2label"]
        self.label2id = checkpoint["label2id"]
        self.W1 = checkpoint["W1"]
        self.b1 = checkpoint["b1"]
        self.W2 = checkpoint["W2"]
        self.b2 = checkpoint["b2"]
        self.model_name = checkpoint.get("model_name", "all-MiniLM-L6-v2")

        # Load pre-trained embedder
        self.embedder = SentenceTransformer(self.model_name)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_shifted = np.exp(logits - np.max(logits))
        return exp_shifted / np.sum(exp_shifted)

    def predict(self, text: str) -> dict:
        """Run text through pre-trained embedding -> 2-layer NN -> Softmax."""
        # 1. Text -> 384-dim Pre-trained Sentence Embedding
        text_vec = self.embedder.encode(text, normalize_embeddings=True)

        # 2. Layer 1 (384 -> 64) + ReLU
        z1 = text_vec @ self.W1 + self.b1
        h1 = np.maximum(0, z1)

        # 3. Layer 2 (64 -> 5) -> Logits -> Softmax Probabilities
        logits = h1 @ self.W2 + self.b2
        probs = self._softmax(logits)

        top_id = int(np.argmax(probs))
        top_label = self.id2label[top_id]
        all_probs = {self.id2label[i]: float(probs[i]) for i in range(len(probs))}

        return {
            "text": text,
            "category": top_label,
            "confidence": float(probs[top_id]),
            "probabilities": all_probs,
        }


def print_prediction(res: dict):
    print(f"\nPrediction:  {res['category'].upper()} ({res['confidence'] * 100:.1f}% confidence)")
    print("Probabilities:")
    for cat, prob in sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"  {cat:<12}: {prob * 100:5.1f}%  {bar}")
    print("-" * 50)


def interactive_loop(classifier: NoteClassifier):
    print("=" * 50)
    print("Live Note Classifier (all-MiniLM-L6-v2 + Neural Network)")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nEnter note: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        res = classifier.predict(user_input)
        print_prediction(res)


if __name__ == "__main__":
    classifier = NoteClassifier("model.pkl")

    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        res = classifier.predict(input_text)
        print(f"\nInput:      \"{res['text']}\"")
        print_prediction(res)
    else:
        interactive_loop(classifier)
