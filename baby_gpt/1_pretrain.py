import os
import pickle
import numpy as np
from tokenizer import QATokenizer
from model import BabyGPT


def pretrain():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pretrain_file = os.path.join(script_dir, "data", "pretrain_corpus.txt")
    sft_file = os.path.join(script_dir, "data", "sft_dialogues.jsonl")

    # Read pre-training raw knowledge
    with open(pretrain_file, "r") as f:
        pretrain_paragraphs = [line.strip() for line in f if line.strip()]

    # Read SFT text to build a unified vocabulary
    import json
    with open(sft_file, "r") as f:
        sft_texts = [json.loads(line)["full_text"] for line in f]

    # 1. Fit Tokenizer on ALL text to create the full English vocabulary
    tokenizer = QATokenizer()
    tokenizer.fit(pretrain_paragraphs + sft_texts)
    vocab_size = len(tokenizer.word2id)

    print("=" * 65)
    print("🚀 STAGE 1: PRE-TRAINING THE BASE MODEL (Raw World Knowledge)")
    print("=" * 65)
    print(f"Vocabulary Size: {vocab_size} tokens")
    print(f"Knowledge Corpus: {len(pretrain_paragraphs)} continuous paragraphs")
    print("Goal: Learn English grammar, concepts, facts, and word relationships.\n")

    # 2. Tokenize raw knowledge into sequences
    encoded_paragraphs = [tokenizer.encode(p) for p in pretrain_paragraphs]
    max_len = max(len(seq) for seq in encoded_paragraphs) + 5

    # 3. Initialize Base Model with random weights
    model = BabyGPT(vocab_size=vocab_size, max_seq_len=max_len, d_model=64, d_ff=128, seed=42)

    # 4. Pre-training Loop (Causal Next-Word Prediction)
    epochs = 60
    lr = 0.02

    # Augment pretrain batches by repeating paragraphs with slight variations
    training_data = []
    for _ in range(30):
        for seq in encoded_paragraphs:
            training_data.append(seq)

    print(f"Pre-training across {len(training_data)} training sequences for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        np.random.shuffle(training_data)
        total_loss = 0.0

        for seq in training_data:
            inputs = seq[:-1]
            targets = seq[1:]

            logits, probs, cache = model.forward(inputs)
            loss = model.compute_loss(probs, targets)
            total_loss += loss

            model.backward(cache, targets, lr=lr)

        avg_loss = total_loss / len(training_data)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | Pre-training Causal Loss: {avg_loss:.4f}")

    # 5. Save the Base Model
    base_ckpt = {
        "model": model,
        "tokenizer": tokenizer,
        "vocab_size": vocab_size,
        "max_seq_len": max_len,
        "stage": "pre-trained-base"
    }
    base_path = os.path.join(script_dir, "base_model.pkl")
    with open(base_path, "wb") as f:
        pickle.dump(base_ckpt, f)

    print(f"\n✓ Base Model successfully saved to '{base_path}'")

    # 6. Test Raw Text Completion
    print("\n--- Testing Base Model: Raw Text Completion ---")
    test_prefixes = [
        "Paris is the capital city of",
        "Butter chicken is a classic North Indian",
        "Progressive resistance training with",
        "FastAPI is a modern high",
        "The S&P 500 index represents"
    ]

    for prefix in test_prefixes:
        p_ids = tokenizer.encode(prefix)
        gen_ids = model.generate(p_ids, max_new_tokens=15, eos_id=-1, temperature=0.0)
        completion = tokenizer.decode(gen_ids, skip_special=False)
        print(f"Prefix:     \"{prefix}\"")
        print(f"Completion: \"{completion}\"\n")


if __name__ == "__main__":
    pretrain()
