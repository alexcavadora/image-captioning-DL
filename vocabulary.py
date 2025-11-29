import json
import pickle
from collections import Counter


class Vocabulary:
    def __init__(self, freq_threshold=4):
        """
        Initialize vocabulary with special tokens

        Args:
            freq_threshold: Minimum word frequency to include in vocabulary
        """
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Add special tokens
        self.pad_token = "<pad>"
        self.start_token = "<start>"
        self.end_token = "<end>"
        self.unk_token = "<unk>"

        # Initialize with special tokens
        for token in [self.pad_token, self.start_token, self.end_token, self.unk_token]:
            self.add_word(token)

    def add_word(self, word):
        """Add a word to the vocabulary"""
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        """Convert word to index, return <unk> if not found"""
        return self.word2idx.get(word, self.word2idx[self.unk_token])

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, caption_file):
        """
        Build vocabulary from COCO captions JSON file (using simple JSON)

        Args:
            caption_file: Path to COCO captions JSON file
        """
        print(f"[INFO] Building vocabulary from {caption_file}")

        # Load JSON file directly (no pycocotools needed!)
        with open(caption_file, "r") as f:
            data = json.load(f)

        annotations = data["annotations"]
        counter = Counter()

        print(f"[INFO] Found {len(annotations)} captions")

        for i, ann in enumerate(annotations):
            caption = str(ann["caption"])
            tokens = caption.lower().split()
            counter.update(tokens)

            if (i + 1) % 10000 == 0:
                print(f"[INFO] Tokenized {i + 1}/{len(annotations)} captions")

        # Add words that meet frequency threshold
        words = [word for word, cnt in counter.items() if cnt >= self.freq_threshold]

        print(f"[INFO] Total words with freq >= {self.freq_threshold}: {len(words)}")

        for word in words:
            self.add_word(word)

        print(f"[INFO] Vocabulary size: {len(self)}")

    def save(self, filepath):
        """Save vocabulary to pickle file"""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] Vocabulary saved to {filepath}")

    @staticmethod
    def load(filepath):
        """Load vocabulary from pickle file"""
        with open(filepath, "rb") as f:
            vocab = pickle.load(f)
        print(f"[INFO] Vocabulary loaded from {filepath}")
        return vocab


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build vocabulary from COCO captions")
    parser.add_argument(
        "--caption-file",
        type=str,
        default="data/annotations/captions_train2014.json",
        help="Path to COCO captions JSON file",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/vocab.pkl", help="Output vocabulary file"
    )
    parser.add_argument(
        "--threshold", type=int, default=1, help="Minimum word frequency threshold"
    )

    args = parser.parse_args()

    vocab = Vocabulary(freq_threshold=args.threshold)
    vocab.build_vocabulary(args.caption_file)
    vocab.save(args.output)
