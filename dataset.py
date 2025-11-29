import json
import os

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class CocoDataset(Dataset):
    def __init__(self, root_dir, caption_file, vocab, transform=None):
        """
        Args:
            root_dir: Directory with COCO images
            caption_file: Path to COCO captions JSON
            vocab: Vocabulary object
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform

        # Load JSON file directly
        print(f"[INFO] Loading captions from {caption_file}")
        with open(caption_file, "r") as f:
            coco_data = json.load(f)

        # Build image_id to filename mapping
        self.image_id_to_filename = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }

        # Store all annotations (caption + image_id pairs)
        self.annotations = coco_data["annotations"]

        print(
            f"[INFO] Loaded {len(self.annotations)} captions for {len(self.image_id_to_filename)} images"
        )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """Get image and caption pair"""
        # Get annotation
        ann = self.annotations[idx]
        caption = ann["caption"]
        img_id = ann["image_id"]

        # Get image filename
        img_filename = self.image_id_to_filename[img_id]
        img_path = os.path.join(self.root_dir, img_filename)

        # Load image
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        tokens = [self.vocab.start_token]
        tokens.extend(caption.lower().split())
        tokens.append(self.vocab.end_token)

        # Convert to indices
        caption_indices = [self.vocab(token) for token in tokens]
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        return image, caption_tensor


class CocoDatasetFast(Dataset):
    """
    Faster version that caches image paths to avoid repeated lookups
    Recommended for training
    """

    def __init__(self, root_dir, caption_file, vocab, transform=None):
        """
        Args:
            root_dir: Directory with COCO images
            caption_file: Path to COCO captions JSON
            vocab: Vocabulary object
            transform: Image transformations
        """
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform

        # Load JSON file directly
        print(f"[INFO] Loading captions from {caption_file}")
        with open(caption_file, "r") as f:
            coco_data = json.load(f)

        # Build image_id to filename mapping
        image_id_to_filename = {
            img["id"]: img["file_name"] for img in coco_data["images"]
        }

        # Store annotations WITH pre-computed image paths
        # This avoids dictionary lookup on every __getitem__ call
        self.data = []
        for ann in coco_data["annotations"]:
            img_filename = image_id_to_filename[ann["image_id"]]
            img_path = os.path.join(root_dir, img_filename)
            self.data.append({"image_path": img_path, "caption": ann["caption"]})

        print(f"[INFO] Loaded {len(self.data)} image-caption pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get image and caption pair"""
        item = self.data[idx]

        # Load image
        image = Image.open(item["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Tokenize caption
        caption = item["caption"]
        tokens = [self.vocab.start_token]
        tokens.extend(caption.lower().split())
        tokens.append(self.vocab.end_token)

        # Convert to indices
        caption_indices = [self.vocab(token) for token in tokens]
        caption_tensor = torch.tensor(caption_indices, dtype=torch.long)

        return image, caption_tensor


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    Pads captions to same length and sorts by caption length (descending)
    Required for pack_padded_sequence

    Args:
        batch: List of (image, caption) tuples

    Returns:
        images: Tensor of images [batch_size, C, H, W]
        captions: Padded captions [batch_size, max_length]
        lengths: Original caption lengths [batch_size]
    """
    # Sort batch by caption length (descending)
    batch.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Get caption lengths
    lengths = torch.tensor([len(cap) for cap in captions], dtype=torch.long)

    # Pad captions to max length in batch
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions_padded, lengths


def get_loader(
    root_dir,
    caption_file,
    vocab,
    transform,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    use_fast=True,
):
    """
    Create DataLoader for COCO dataset

    Args:
        root_dir: Image directory
        caption_file: COCO captions JSON
        vocab: Vocabulary object
        transform: Image transformations
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker threads
        use_fast: Use CocoDatasetFast (recommended)

    Returns:
        DataLoader with custom collate function
    """
    # Choose dataset implementation
    if use_fast:
        dataset = CocoDatasetFast(root_dir, caption_file, vocab, transform)
    else:
        dataset = CocoDataset(root_dir, caption_file, vocab, transform)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return data_loader


# Utility function to check dataset
def test_dataset(root_dir, caption_file, vocab, transform=None):
    """
    Test dataset loading and print statistics
    """
    from torchvision import transforms

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )

    print("\n" + "=" * 60)
    print("Testing COCO Dataset")
    print("=" * 60)

    # Test both versions
    print("\n1. Testing CocoDataset...")
    dataset1 = CocoDataset(root_dir, caption_file, vocab, transform)
    print(f"   Dataset size: {len(dataset1)}")

    # Load first sample
    img, cap = dataset1[0]
    print(f"   Sample image shape: {img.shape}")
    print(f"   Sample caption length: {len(cap)}")
    print(f"   Sample caption (indices): {cap[:10].tolist()}...")

    # Convert caption to words
    words = [vocab.idx2word[idx.item()] for idx in cap]
    print(f"   Sample caption (words): {' '.join(words)}")

    print("\n2. Testing CocoDatasetFast...")
    dataset2 = CocoDatasetFast(root_dir, caption_file, vocab, transform)
    print(f"   Dataset size: {len(dataset2)}")

    img2, cap2 = dataset2[0]
    print(f"   Sample image shape: {img2.shape}")
    print(f"   Sample caption length: {len(cap2)}")

    print("\n3. Testing DataLoader...")
    loader = get_loader(
        root_dir,
        caption_file,
        vocab,
        transform,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, captions, lengths = next(iter(loader))
    print(f"   Batch images shape: {images.shape}")
    print(f"   Batch captions shape: {captions.shape}")
    print(f"   Batch lengths: {lengths.tolist()}")
    print(
        f"   Captions sorted by length: {all(lengths[i] >= lengths[i + 1] for i in range(len(lengths) - 1))}"
    )

    print("\n" + "=" * 60)
    print("✓ Dataset test complete!")
    print("=" * 60 + "\n")

    return dataset2, loader


if __name__ == "__main__":
    """
    Test the dataset implementation
    """
    import sys

    sys.path.append("..")
    import torchvision.transforms as transforms

    from vocabulary import Vocabulary

    # Load vocabulary
    vocab = Vocabulary.load("outputs/vocab.pkl")

    # Define transform
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Test dataset
    dataset, loader = test_dataset(
        root_dir="data/train2014",
        caption_file="data/annotations/captions_train2014.json",
        vocab=vocab,
        transform=transform,
    )
