import json
import re

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "is", "was", "are", "were", "to", "for",
    "in", "on", "at", "of", "with", "it", "this", "that", "my", "i", "me", "we", "our",
    "you", "your", "he", "she", "they", "them", "his", "her", "their", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "personal", "log", "day", "note", "reminder",
    "quick", "observation", "reflection", "update", "week", "just", "so", "too", "very",
    ".", ",", "-", "—", ":", "!", "?", ";", "(", ")", "[", "]", "\"", "'"
}


class SimpleTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.word2id = {self.pad_token: 0, self.unk_token: 1}
        self.id2word = {0: self.pad_token, 1: self.unk_token}

    def tokenize(self, text: str) -> list[str]:
        """Split text into lowercase words, filtering out filler stopwords and punctuation."""
        words = re.findall(r"\w+", text.lower())
        return [w for w in words if w not in STOPWORDS]

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary from a list of text strings."""
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.word2id:
                    new_id = len(self.word2id)
                    self.word2id[token] = new_id
                    self.id2word[new_id] = token

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        tokens = self.tokenize(text)
        return [self.word2id.get(token, self.word2id[self.unk_token]) for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Convert token IDs back into token words."""
        return [self.id2word.get(idx, self.unk_token) for idx in ids]


if __name__ == "__main__":
    with open("train.jsonl") as f:
        dataset = [json.loads(line) for line in f]
    texts = [item["text"] for item in dataset]

    tokenizer = SimpleTokenizer()
    tokenizer.fit(texts)

    print(f"Vocabulary size: {len(tokenizer.word2id)} tokens\n")

    sample_text = dataset[0]["text"]
    tokens = tokenizer.tokenize(sample_text)
    token_ids = tokenizer.encode(sample_text)
    decoded_tokens = tokenizer.decode(token_ids)

    print(f"Sample text from dataset:\n  \"{sample_text}\"\n")
    print(f"Tokens ({len(tokens)}):\n  {tokens}\n")
    print(f"Token IDs:\n  {token_ids}\n")
    print(f"Decoded:\n  {decoded_tokens}")
