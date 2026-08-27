import re
import collections


class QATokenizer:
    def __init__(self, max_vocab_size: int = 5000, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq

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
        """Build vocabulary from texts, filtering one-off OCR typos."""
        counter = collections.Counter()
        for text in texts:
            for tok in self.tokenize(text):
                if tok not in self.special_tokens:
                    counter[tok] += 1

        # Keep words that meet min_freq, sorted by frequency
        most_common = counter.most_common(self.max_vocab_size - len(self.special_tokens))

        for tok, count in most_common:
            if count >= self.min_freq and tok not in self.word2id:
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

        text = " ".join(words)
        text = re.sub(r"\s+([.,!?:;])", r"\1", text)
        return text.strip()
