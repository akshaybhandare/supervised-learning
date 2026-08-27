import os
import sys
import pickle
import numpy as np


class BabyGPTChat:
    def __init__(self, model_path: str = None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "baby_gpt.pkl")

        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.model = checkpoint["model"]
        self.tokenizer = checkpoint["tokenizer"]
        self.eos_id = self.tokenizer.word2id["<eos>"]

    def answer_question(self, question: str, temperature: float = 0.0) -> str:
        """Format question into prompt, run autoregressive generation, and return answer."""
        prompt = f"<bos> Q: {question.strip()} A:"
        prompt_ids = self.tokenizer.encode(prompt)

        # Autoregressive generation
        generated_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=30,
            eos_id=self.eos_id,
            temperature=temperature
        )

        # Extract only the newly generated answer tokens
        answer_ids = generated_ids[len(prompt_ids):]
        raw_answer = self.tokenizer.decode(answer_ids, skip_special=True)
        return raw_answer.strip()


def interactive_loop():
    print("=" * 60)
    print("🤖 Baby-GPT Q&A Language Model (Pure NumPy Transformer)")
    print("Ask questions about Travel, Food, Health, Tech, or Finance!")
    print("Type 'quit', 'exit', or 'q' to stop")
    print("=" * 60)

    bot = BabyGPTChat()

    while True:
        try:
            user_input = input("\nAsk a question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        answer = bot.answer_question(user_input, temperature=0.0)
        print(f"Baby-GPT: {answer}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        bot = BabyGPTChat()
        ans = bot.answer_question(q, temperature=0.0)
        print(f"\nQuestion: {q}")
        print(f"Baby-GPT: {ans}\n")
    else:
        interactive_loop()
