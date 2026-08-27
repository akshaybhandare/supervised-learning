import os
import json
import pickle
import numpy as np


def post_train_sft():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_ckpt_path = os.path.join(script_dir, "base_model.pkl")

    if not os.path.exists(base_ckpt_path):
        print("Please run '1_pretrain.py' first to produce 'base_model.pkl'!")
        return

    # 1. Load Pre-trained Base Model
    with open(base_ckpt_path, "rb") as f:
        base_ckpt = pickle.load(f)

    model = base_ckpt["model"]
    tokenizer = base_ckpt["tokenizer"]

    print("=" * 65)
    print("🎓 STAGE 2: POST-TRAINING / SUPERVISED FINE-TUNING (SFT)")
    print("=" * 65)
    print("Goal: Teach the knowledgeable Base Model how to act as a conversational Chatbot.")
    print("Starting from pre-trained weights in 'base_model.pkl'...\n")

    # 2. Load SFT Dialogues
    sft_file = os.path.join(script_dir, "data", "sft_dialogues.jsonl")
    with open(sft_file, "r") as f:
        sft_data = [json.loads(line)["full_text"] for line in f]

    encoded_dialogues = [tokenizer.encode(d) for d in sft_data]

    # 3. Fine-tuning Loop (Lower learning rate to adapt behavior without catastrophic forgetting)
    epochs = 35
    sft_lr = 0.008

    print(f"Fine-tuning across {len(encoded_dialogues)} dialogue examples for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        np.random.shuffle(encoded_dialogues)
        total_loss = 0.0

        for seq in encoded_dialogues:
            inputs = seq[:-1]
            targets = seq[1:]

            logits, probs, cache = model.forward(inputs)
            loss = model.compute_loss(probs, targets)
            total_loss += loss

            model.backward(cache, targets, lr=sft_lr)

        avg_loss = total_loss / len(encoded_dialogues)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs} | SFT Dialogue Loss: {avg_loss:.4f}")

    # 4. Save Final Fine-Tuned Chat Model
    chat_ckpt = {
        "model": model,
        "tokenizer": tokenizer,
        "vocab_size": base_ckpt["vocab_size"],
        "max_seq_len": base_ckpt["max_seq_len"],
        "stage": "sft-chat-model"
    }
    chat_path = os.path.join(script_dir, "chat_model.pkl")
    with open(chat_path, "wb") as f:
        pickle.dump(chat_ckpt, f)

    print(f"\n✓ Fine-Tuned Chat Model successfully saved to '{chat_path}'")


if __name__ == "__main__":
    post_train_sft()
