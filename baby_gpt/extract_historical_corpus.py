import os
import tarfile
import json
import re


def clean_article_text(text: str) -> str:
    """Clean OCR hyphens, broken newlines, and irregular whitespace."""
    # Fix hyphenated words broken across lines: e.g. "manslaugh-\nter" -> "manslaughter"
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Replace newlines with spaces
    text = text.replace("\n", " ")
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    # Strip strange non-ASCII OCR artifacts while keeping standard punctuation
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def extract_paragraphs_from_tar(tar_path: str, max_paragraphs: int = 2500) -> list[str]:
    """Stream JSON files directly from tar.gz and extract clean paragraphs."""
    paragraphs = []
    if not os.path.exists(tar_path):
        print(f"File {tar_path} does not exist.")
        return paragraphs

    print(f"Streaming through '{os.path.basename(tar_path)}'...")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.name.endswith(".json"):
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = json.load(f)

                # Extract from full articles
                articles = data.get("full articles", [])
                for art in articles:
                    body = art.get("article", "")
                    cleaned = clean_article_text(body)
                    # Filter out short or noisy fragments
                    if len(cleaned) >= 100:
                        # Split long articles into manageable chunks of 30-60 words
                        words = cleaned.split()
                        chunk_size = 40
                        for i in range(0, len(words), chunk_size):
                            chunk = " ".join(words[i : i + chunk_size])
                            if len(chunk.split()) >= 15:
                                paragraphs.append(chunk)
                                if len(paragraphs) >= max_paragraphs:
                                    return paragraphs

            except Exception:
                continue

    return paragraphs


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")

    tar_1826 = os.path.join(data_dir, "faro_1826.tar.gz")
    tar_1927 = os.path.join(data_dir, "faro_1927.tar.gz")

    # Extract 2,500 paragraphs from 1826 and 2,500 paragraphs from 1927
    p_1826 = extract_paragraphs_from_tar(tar_1826, max_paragraphs=2500)
    p_1927 = extract_paragraphs_from_tar(tar_1927, max_paragraphs=2500)

    combined_historical = p_1826 + p_1927
    print(f"\n✓ Extracted {len(p_1826)} paragraphs from 1826")
    print(f"✓ Extracted {len(p_1927)} paragraphs from 1927")
    print(f"✓ Total historical pre-training paragraphs: {len(combined_historical)}")

    output_path = os.path.join(data_dir, "pretrain_corpus_historical.txt")
    with open(output_path, "w") as f:
        for p in combined_historical:
            f.write(p + "\n")

    print(f"✓ Saved clean historical corpus to '{output_path}'")

    # Also update pretrain_corpus.txt with a rich blend of historical + core facts
    modern_pretrain_path = os.path.join(data_dir, "pretrain_corpus.txt")
    modern_paragraphs = []
    if os.path.exists(modern_pretrain_path):
        with open(modern_pretrain_path, "r") as f:
            modern_paragraphs = [line.strip() for line in f if line.strip()]

    blended_corpus = modern_paragraphs + combined_historical
    with open(modern_pretrain_path, "w") as f:
        for p in blended_corpus:
            f.write(p + "\n")

    print(f"✓ Updated '{modern_pretrain_path}' with {len(blended_corpus)} total pre-training paragraphs!")


if __name__ == "__main__":
    main()
