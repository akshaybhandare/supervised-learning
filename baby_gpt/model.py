import numpy as np


class BabyGPT:
    def __init__(self, vocab_size: int, max_seq_len: int = 64, d_model: int = 64, d_ff: int = 128, seed: int = 42):
        np.random.seed(seed)
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.d_ff = d_ff

        # 1. Embeddings: Token & Positional
        self.tok_embed = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_embed = np.random.randn(max_seq_len, d_model) * 0.02

        # 2. Causal Self-Attention Weights
        scale = 1.0 / np.sqrt(d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.b_q = np.zeros(d_model)

        self.W_k = np.random.randn(d_model, d_model) * scale
        self.b_k = np.zeros(d_model)

        self.W_v = np.random.randn(d_model, d_model) * scale
        self.b_v = np.zeros(d_model)

        self.W_o = np.random.randn(d_model, d_model) * scale
        self.b_o = np.zeros(d_model)

        # 3. FeedForward Network Weights (MLP)
        self.W_ff1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b_ff1 = np.zeros(d_ff)

        self.W_ff2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b_ff2 = np.zeros(d_model)

        # 4. Output Head (d_model -> vocab_size)
        self.W_head = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
        self.b_head = np.zeros(vocab_size)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_shifted = x - np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def forward(self, input_ids: list[int]):
        """
        Forward pass for a sequence of token IDs of length T:
          input_ids -> Token+Pos Embed -> Causal Self-Attention -> FFN -> Logits (T x vocab_size)
        """
        T = len(input_ids)
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"

        # 1. Embeddings
        x_tok = self.tok_embed[input_ids]               # (T, d_model)
        x_pos = self.pos_embed[:T]                       # (T, d_model)
        x0 = x_tok + x_pos                               # (T, d_model)

        # 2. Causal Self-Attention
        Q = x0 @ self.W_q + self.b_q                     # (T, d_model)
        K = x0 @ self.W_k + self.b_k                     # (T, d_model)
        V = x0 @ self.W_v + self.b_v                     # (T, d_model)

        scores = (Q @ K.T) / np.sqrt(self.d_model)       # (T, T)

        # Causal Mask (upper triangle cannot be seen)
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[causal_mask] = -1e9

        attn_weights = self._softmax(scores, axis=-1)   # (T, T)
        context = attn_weights @ V                       # (T, d_model)
        attn_out = context @ self.W_o + self.b_o         # (T, d_model)

        x1 = x0 + attn_out                               # Residual connection (T, d_model)

        # 3. FeedForward Network
        z_ff = x1 @ self.W_ff1 + self.b_ff1              # (T, d_ff)
        h_ff = np.maximum(0, z_ff)                       # ReLU (T, d_ff)
        ff_out = h_ff @ self.W_ff2 + self.b_ff2          # (T, d_model)

        x2 = x1 + ff_out                                 # Residual connection (T, d_model)

        # 4. Output Head
        logits = x2 @ self.W_head + self.b_head          # (T, vocab_size)
        probs = self._softmax(logits, axis=-1)

        cache = {
            "T": T, "input_ids": input_ids, "x0": x0, "Q": Q, "K": K, "V": V,
            "scores": scores, "attn_weights": attn_weights, "context": context,
            "x1": x1, "z_ff": z_ff, "h_ff": h_ff, "x2": x2, "logits": logits, "probs": probs
        }
        return logits, probs, cache

    def compute_loss(self, probs: np.ndarray, targets: list[int]) -> float:
        """Cross-entropy loss over the shifted sequence (predicting t+1 from t)."""
        T = len(targets)
        # probs has shape (T, vocab_size), targets has length T
        correct_log_probs = -np.log(probs[np.arange(T), targets] + 1e-9)
        return float(np.mean(correct_log_probs))

    def backward(self, cache: dict, targets: list[int], lr: float = 0.01):
        """Analytical backpropagation through all layers and parameter updates."""
        T = cache["T"]
        input_ids = cache["input_ids"]
        probs = cache["probs"]

        # Target sequence is length T-1 (predicting next token)
        # Gradient of cross-entropy w.r.t logits (T, vocab_size)
        dlogits = probs.copy()
        dlogits[np.arange(T), targets] -= 1.0
        dlogits /= T

        # Gradients for Output Head
        dW_head = cache["x2"].T @ dlogits
        db_head = np.sum(dlogits, axis=0)
        dx2 = dlogits @ self.W_head.T                    # (T, d_model)

        # Gradients for FFN
        dh_ff = dx2 @ self.W_ff2.T                       # (T, d_ff)
        dz_ff = dh_ff * (cache["z_ff"] > 0)              # ReLU backward
        dW_ff2 = cache["h_ff"].T @ dx2
        db_ff2 = np.sum(dx2, axis=0)
        dW_ff1 = cache["x1"].T @ dz_ff
        db_ff1 = np.sum(dz_ff, axis=0)
        dx1 = dx2 + dz_ff @ self.W_ff1.T                 # Include residual connection

        # Gradients for Attention Output Projection
        dcontext = dx1 @ self.W_o.T                      # (T, d_model)
        dW_o = cache["context"].T @ dx1
        db_o = np.sum(dx1, axis=0)

        # Gradients for Attention Matrix & Values
        dV = cache["attn_weights"].T @ dcontext          # (T, d_model)
        dattn = dcontext @ cache["V"].T                  # (T, T)

        # Softmax backward
        # dscores = attn * (dattn - sum(dattn * attn, axis=-1, keepdims=True))
        sum_dattn_attn = np.sum(dattn * cache["attn_weights"], axis=-1, keepdims=True)
        dscores = cache["attn_weights"] * (dattn - sum_dattn_attn)
        dscores /= np.sqrt(self.d_model)

        # Gradients for Q and K
        dQ = dscores @ cache["K"]                        # (T, d_model)
        dK = dscores.T @ cache["Q"]                      # (T, d_model)

        # Gradients for Q, K, V linear projections
        dW_q = cache["x0"].T @ dQ
        db_q = np.sum(dQ, axis=0)
        dW_k = cache["x0"].T @ dK
        db_k = np.sum(dK, axis=0)
        dW_v = cache["x0"].T @ dV
        db_v = np.sum(dV, axis=0)

        # Gradient at x0 (adding residual dx1)
        dx0 = dx1 + dQ @ self.W_q.T + dK @ self.W_k.T + dV @ self.W_v.T

        # Update Embeddings
        for t, tok_id in enumerate(input_ids):
            self.tok_embed[tok_id] -= lr * dx0[t]
            self.pos_embed[t] -= lr * dx0[t]

        # Update Weights
        self.W_q -= lr * dW_q
        self.b_q -= lr * db_q
        self.W_k -= lr * dW_k
        self.b_k -= lr * db_k
        self.W_v -= lr * dW_v
        self.b_v -= lr * db_v
        self.W_o -= lr * dW_o
        self.b_o -= lr * db_o

        self.W_ff1 -= lr * dW_ff1
        self.b_ff1 -= lr * db_ff1
        self.W_ff2 -= lr * dW_ff2
        self.b_ff2 -= lr * db_ff2

        self.W_head -= lr * dW_head
        self.b_head -= lr * db_head

    def generate(self, prompt_ids: list[int], max_new_tokens: int = 25, eos_id: int = 3, temperature: float = 0.5) -> list[int]:
        """Autoregressive generation loop: predict next token one by one until eos or limit."""
        curr_ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            if len(curr_ids) >= self.max_seq_len:
                break

            # Forward pass on current prefix
            logits, probs, _ = self.forward(curr_ids)

            # Get logits for the VERY LAST token position
            last_logits = logits[-1]

            # Temperature scaling
            if temperature > 0:
                scaled_logits = last_logits / temperature
                p = self._softmax(scaled_logits)
                # Sample from top probabilities
                next_token = int(np.random.choice(len(p), p=p))
            else:
                next_token = int(np.argmax(last_logits))

            curr_ids.append(next_token)

            if next_token == eos_id:
                break

        return curr_ids
