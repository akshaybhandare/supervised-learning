import os
import pickle
import numpy as np


def test_base_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_path = os.path.join(script_dir, "base_model.pkl")

    if not os.path.exists(ckpt_path):
        print("Please run '1_pretrain.py' first to train the Base Model!")
        return

    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    model = ckpt["model"]
    tokenizer = ckpt["tokenizer"]

    print("=" * 65)
    print("🔍 INSPECTING THE BASE MODEL (Before Fine-Tuning / SFT)")
    print("=" * 65)
    print("A Base Model has learned world knowledge, but DOES NOT know how to chat.\n")

    print("--- Test 1: Completing Raw Knowledge Sentences (Works Great!) ---")
    raw_prompts = [
        "Tokyo is the bustling capital of",
        "Neapolitan pizza is baked in a",
        "A barbell bench press of 225 pounds",
        "Docker containers package application code and",
        "A Roth IRA is an individual retirement"
    ]

    for p in raw_prompts:
        p_ids = tokenizer.encode(p)
        gen_ids = model.generate(p_ids, max_new_tokens=12, eos_id=-1, temperature=0.0)
        output = tokenizer.decode(gen_ids, skip_special=True)
        print(f"Input:  \"{p}\"")
        print(f"Output: \"{output}\"\n")

    print("=" * 65)
    print("--- Test 2: Asking the Base Model a Chat Question (Fails / Rambles!) ---")
    chat_prompts = [
        "<bos> User: Where did you stay in Paris? Assistant:",
        "<bos> User: What is the S&P 500? Assistant:"
    ]

    for p in chat_prompts:
        p_ids = tokenizer.encode(p)
        gen_ids = model.generate(p_ids, max_new_tokens=15, eos_id=tokenizer.word2id["<eos>"], temperature=0.0)
        output = tokenizer.decode(gen_ids, skip_special=False)
        print(f"Prompt: \"{p}\"")
        print(f"Output: \"{output}\"")
        print(">> Note: The Base Model repeats words or tries to continue paragraphs because it was never trained on 'User:' / 'Assistant:' conversation protocols!\n")


if __name__ == "__main__":
    test_base_model()
