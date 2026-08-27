import re


class QATokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.word2id = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.id2word = {idx: token for idx, token in enumerate(self.special_tokens)}

    def tokenize(self, text: str) -> list[str]:
        """Tokenize string while keeping special tags <bos>, <eos>, <pad>, <unk> intact."""
        pattern = r"(<bos>|<eos>|<pad>|<unk>|\w+|[^\w\s])"
        raw_tokens = re.findall(pattern, text)
        result = []
        for tok in raw_tokens:
            if tok in self.special_tokens:
                result.append(tok)
            else:
                result.append(tok.lower())
        return result

    def fit(self, texts: list[str]) -> None:
        """Build vocabulary from texts."""
        for text in texts:
            for tok in self.tokenize(text):
                if tok not in self.word2id:
                    new_id = len(self.word2id)
                    self.word2id[tok] = new_id
                    self.id2word[new_id] = tok

    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        tokens = self.tokenize(text)
        return [self.word2id.get(tok, self.word2id[self.unk_token]) for tok in tokens]

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Convert token IDs back into human readable text."""
        words = []
        for idx in ids:
            tok = self.id2word.get(idx, self.unk_token)
            if skip_special and tok in self.special_tokens:
                continue
            words.append(tok)

        # Clean spacing for punctuation
        text = " ".join(words)
        text = re.sub(r"\s+([.,!?:;])", r"\1", text)
        return text.strip()


if __name__ == "__main__":
    import json
    with open("qa_data.jsonl") as f:
        data = [json.loads(line)["text"] for line in f]

    tok = QATokenizer()
    tok.fit(data)
    print(f"Vocabulary size: {len(tok.word2id)} tokens")

    sample = data[0]
    encoded = tok.encode(sample)
    decoded = tok.decode(encoded, skip_special=False)
    print(f"\nOriginal: {sample}")
    print(f"Token IDs ({len(encoded)}): {encoded[:8]} ...")
    print(f"Decoded:  {decoded}")
