import os
import json
import pickle
import numpy as np
from tokenizer import QATokenizer
from model import BabyGPT


def load_dataset(filepath: str) -> list[str]:
    with open(filepath, "r") as f:
        return [json.loads(line)["text"] for line in f]


def train():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "qa_data.jsonl")

    raw_texts = load_dataset(data_path)

    # 1. Fit Tokenizer
    tokenizer = QATokenizer()
    tokenizer.fit(raw_texts)
    vocab_size = len(tokenizer.word2id)
    print(f"=== Baby-GPT Q&A Language Model (Pure NumPy) ===")
    print(f"Vocabulary Size: {vocab_size} tokens | Training Examples: {len(raw_texts)}")

    # 2. Encode all training sequences
    encoded_seqs = [tokenizer.encode(t) for t in raw_texts]
    max_len = max(len(seq) for seq in encoded_seqs) + 5
    print(f"Max sequence length in dataset: {max_len}")

    # 3. Initialize Model
    model = BabyGPT(vocab_size=vocab_size, max_seq_len=max_len, d_model=64, d_ff=128)

    epochs = 40
    lr = 0.02

    test_prompts = [
        "<bos> Q: Where did you stay in Paris? A:",
        "<bos> Q: What did you cook for Sunday dinner? A:",
        "<bos> Q: What is your bench press record? A:",
        "<bos> Q: What framework did you use for the API? A:",
        "<bos> Q: What index fund do you invest in? A:"
    ]

    print("\nStarting training loop...")
    for epoch in range(1, epochs + 1):
        # Shuffle dataset
        np.random.shuffle(encoded_seqs)
        total_loss = 0.0

        for seq in encoded_seqs:
            # Inputs: [t_0, t_1, ... t_{n-1}]
            # Targets: [t_1, t_2, ... t_n]
            inputs = seq[:-1]
            targets = seq[1:]

            # Forward
            logits, probs, cache = model.forward(inputs)
            loss = model.compute_loss(probs, targets)
            total_loss += loss

            # Backward + update
            model.backward(cache, targets, lr=lr)

        avg_loss = total_loss / len(encoded_seqs)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Cross-Entropy Loss: {avg_loss:.4f}")

            # Sample generation
            sample_prompt = test_prompts[(epoch // 10) % len(test_prompts)]
            prompt_ids = tokenizer.encode(sample_prompt)
            gen_ids = model.generate(prompt_ids, max_new_tokens=20, eos_id=tokenizer.word2id["<eos>"], temperature=0.0)
            gen_text = tokenizer.decode(gen_ids, skip_special=False)
            print(f"   Sample Output: {gen_text}\n")

    # 4. Save Model Checkpoint
    checkpoint = {
        "model": model,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "max_seq_len": max_len,
    }
    ckpt_path = os.path.join(script_dir, "baby_gpt.pkl")
    with open(ckpt_path, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"✓ Model successfully saved to '{ckpt_path}'")


if __name__ == "__main__":
    train()
