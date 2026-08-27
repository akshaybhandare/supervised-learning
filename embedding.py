import json
import numpy as np
from tokenizer import SimpleTokenizer


class SimpleEmbedding:
    def __init__(self, vocab_size: int, embed_dim: int = 16, seed: int = 42):
        np.random.seed(seed)
        # Random matrix of shape: (vocab_size, embed_dim)
        self.matrix = np.random.randn(vocab_size, embed_dim)
        # Pad (0) and Unk (1) must be neutral zeros so unknown words don't inject random bias
        if vocab_size > 0:
            self.matrix[0] = 0.0
        if vocab_size > 1:
            self.matrix[1] = 0.0

    def lookup(self, token_ids: list[int]) -> np.ndarray:
        """Lookup embedding vectors for a list of token IDs. Returns shape (seq_len, embed_dim)."""
        if not token_ids:
            return np.zeros((0, self.matrix.shape[1]))
        return self.matrix[token_ids]

    def pool(self, vectors: np.ndarray, method: str = "mean") -> np.ndarray:
        """Combine token vectors into a single text embedding vector of shape (embed_dim,)."""
        if len(vectors) == 0:
            return np.zeros(self.matrix.shape[1])
        if method == "mean":
            return np.mean(vectors, axis=0)
        elif method == "max":
            return np.max(vectors, axis=0)
        elif method == "sum":
            return np.sum(vectors, axis=0)
        else:
            raise ValueError(f"Unknown pooling method '{method}'. Supported: 'mean', 'max', 'sum'")

    def embed_text(self, text: str, tokenizer: SimpleTokenizer, pooling: str = "mean") -> np.ndarray:
        """End-to-end: text -> token IDs -> token embeddings -> pooled text embedding (embed_dim,)."""
        token_ids = tokenizer.encode(text)
        vectors = self.lookup(token_ids)
        return self.pool(vectors, method=pooling)

    def embed_batch(self, texts: list[str], tokenizer: SimpleTokenizer, pooling: str = "mean") -> np.ndarray:
        """Batch helper: list of texts -> matrix of shape (num_texts, embed_dim)."""
        return np.array([self.embed_text(t, tokenizer, pooling=pooling) for t in texts])


if __name__ == "__main__":
    # 1. Load dataset & fit tokenizer
    with open("train.jsonl") as f:
        dataset = [json.loads(line) for line in f]
    texts = [item["text"] for item in dataset]

    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)

    # 2. Create embedding layer
    vocab_size = len(tokenizer.word2id)  # 3031
    embed_dim = 8                         # Small dimension for clear display
    embedding = SimpleEmbedding(vocab_size=vocab_size, embed_dim=embed_dim)

    # 3. Demonstrate individual token lookup
    sample_text = dataset[0]["text"]
    token_ids = tokenizer.encode(sample_text)
    token_vectors = embedding.lookup(token_ids)

    print(f"Sample text: \"{sample_text}\"\n")
    print(f"Token vectors shape:        {token_vectors.shape}  (19 tokens x 8 dim)")

    # 4. Demonstrate pooling helper methods
    mean_vec = embedding.pool(token_vectors, method="mean")
    max_vec = embedding.pool(token_vectors, method="max")
    print(f"Single text embedding (mean): {mean_vec.shape} -> {mean_vec.round(3)}")
    print(f"Single text embedding (max):  {max_vec.shape} -> {max_vec.round(3)}\n")

    # 5. Demonstrate end-to-end embed_text and batch embedding
    direct_vec = embedding.embed_text(sample_text, tokenizer, pooling="mean")
    batch_vecs = embedding.embed_batch(texts[:5], tokenizer, pooling="mean")
    print(f"Direct embed_text() shape:    {direct_vec.shape}")
    print(f"Batch embed_batch() shape:    {batch_vecs.shape}  (5 notes x 8 dim)")
