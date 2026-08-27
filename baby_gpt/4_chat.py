import os
import sys
import pickle


class BabyGPTAssistant:
    def __init__(self, model_path: str = None):
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "chat_model.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint '{model_path}' not found. Please run 3_post_train.py first!")

        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.model = checkpoint["model"]
        self.tokenizer = checkpoint["tokenizer"]
        self.eos_id = self.tokenizer.word2id["<eos>"]

    def reply(self, user_message: str, temperature: float = 0.0) -> str:
        """Format user message into chat prompt, generate answer, and return clean response."""
        prompt = f"<bos> User: {user_message.strip()} Assistant:"
        prompt_ids = self.tokenizer.encode(prompt)

        generated_ids = self.model.generate(
            prompt_ids,
            max_new_tokens=30,
            eos_id=self.eos_id,
            temperature=temperature
        )

        answer_ids = generated_ids[len(prompt_ids):]
        raw_answer = self.tokenizer.decode(answer_ids, skip_special=True)
        return raw_answer.strip()


def interactive_chat():
    print("=" * 65)
    print("🤖 Baby-GPT Chat Model (Post-Trained / SFT)")
    print("Ask questions about Travel, Food, Health, Tech, or Finance!")
    print("Type 'quit', 'exit', or 'q' to exit")
    print("=" * 65)

    assistant = BabyGPTAssistant()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        response = assistant.reply(user_input, temperature=0.0)
        print(f"Baby-GPT: {response}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        msg = " ".join(sys.argv[1:])
        assistant = BabyGPTAssistant()
        ans = assistant.reply(msg)
        print(f"\nYou: {msg}")
        print(f"Baby-GPT: {ans}\n")
    else:
        interactive_chat()
