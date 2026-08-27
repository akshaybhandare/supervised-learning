import os
import sys
import json
import pickle
import warnings
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from sentence_transformers import SentenceTransformer


# ----------------------------------------------------------------------
# 1. Load Data
# ----------------------------------------------------------------------
def load_jsonl(filepath: str) -> list[dict]:
    with open(filepath, "r") as f:
        return [json.loads(line) for line in f]


train_data = load_jsonl("train.jsonl")
eval_data = load_jsonl("eval.jsonl")

train_texts = [item["text"] for item in train_data]
train_labels = [item["label"] for item in train_data]

eval_texts = [item["text"] for item in eval_data]
eval_labels = [item["label"] for item in eval_data]


# ----------------------------------------------------------------------
# 2. Labels -> Numeric IDs
# ----------------------------------------------------------------------
LABELS = sorted(list(set(train_labels)))
label2id = {label: idx for idx, label in enumerate(LABELS)}
id2label = {idx: label for label, idx in label2id.items()}

y_train = np.array([label2id[label] for label in train_labels])
y_eval = np.array([label2id[label] for label in eval_labels])


# ----------------------------------------------------------------------
# 3. Pre-trained Sentence Transformer Embeddings (all-MiniLM-L6-v2)
# ----------------------------------------------------------------------
print("Loading pre-trained embedding model 'all-MiniLM-L6-v2'...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding train and eval notes into dense 384-dimensional vectors...")
X_train = embedder.encode(train_texts, show_progress_bar=False, normalize_embeddings=True)
X_eval = embedder.encode(eval_texts, show_progress_bar=False, normalize_embeddings=True)

EMBED_DIM = X_train.shape[1]  # 384
HIDDEN_DIM = 64


# ----------------------------------------------------------------------
# 4. Helper Functions & Activations
# ----------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    return dout * (x > 0)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp_shifted = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)


# ----------------------------------------------------------------------
# 5. Central Training Loop (2-Layer NN with Mini-batch SGD)
# ----------------------------------------------------------------------
def train():
    num_classes = len(LABELS)

    # Initialize weights: 384 (Pretrained) -> 64 (Hidden) -> 5 (Classes)
    np.random.seed(42)
    W1 = np.random.randn(EMBED_DIM, HIDDEN_DIM) * np.sqrt(2.0 / EMBED_DIM)
    b1 = np.zeros(HIDDEN_DIM)

    W2 = np.random.randn(HIDDEN_DIM, num_classes) * np.sqrt(2.0 / HIDDEN_DIM)
    b2 = np.zeros(num_classes)

    epochs = 40
    learning_rate = 0.05
    batch_size = 32

    print(f"\n=== Training 2-Layer NN: {EMBED_DIM} (Input) -> {HIDDEN_DIM} (Hidden) -> {num_classes} (Classes) ===")
    print(f"Dataset: {len(X_train)} train notes | Classes: {num_classes}\n")

    num_samples = len(X_train)

    for epoch in range(1, epochs + 1):
        # Shuffle batches
        indices = np.random.permutation(num_samples)
        total_loss = 0.0
        correct = 0

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            b_size = len(y_batch)

            # 1. Forward Pass
            z1 = X_batch @ W1 + b1
            h1 = relu(z1)
            logits = h1 @ W2 + b2
            probs = softmax(logits)

            # 2. Cross-entropy loss
            loss = -np.mean(np.log(probs[np.arange(b_size), y_batch] + 1e-9))
            total_loss += loss * b_size

            preds = np.argmax(probs, axis=1)
            correct += np.sum(preds == y_batch)

            # 3. Backward Pass
            dlogits = probs.copy()
            dlogits[np.arange(b_size), y_batch] -= 1.0
            dlogits /= b_size

            dW2 = h1.T @ dlogits
            db2 = np.sum(dlogits, axis=0)

            dh1 = dlogits @ W2.T
            dz1 = relu_backward(dh1, z1)

            dW1 = X_batch.T @ dz1
            db1 = np.sum(dz1, axis=0)

            # 4. Update Weights
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1

        # Metrics
        train_loss = total_loss / num_samples
        train_acc = correct / num_samples

        # Eval pass
        eval_z1 = X_eval @ W1 + b1
        eval_h1 = relu(eval_z1)
        eval_logits = eval_h1 @ W2 + b2
        eval_probs = softmax(eval_logits)
        eval_preds = np.argmax(eval_probs, axis=1)
        eval_acc = np.mean(eval_preds == y_eval)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.1f}% | Eval Acc: {eval_acc * 100:.1f}%")

    return W1, b1, W2, b2


# ----------------------------------------------------------------------
# 6. Save Model Function
# ----------------------------------------------------------------------
def save_model(filepath: str, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray):
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)

    checkpoint = {
        "model_name": "all-MiniLM-L6-v2",
        "id2label": id2label,
        "label2id": label2id,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
    }
    with open(filepath, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"\n✓ Model successfully saved to '{filepath}'")


# ----------------------------------------------------------------------
# 7. Execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    W1, b1, W2, b2 = train()
    save_model("model.pkl", W1, b1, W2, b2)
