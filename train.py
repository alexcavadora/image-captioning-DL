import argparse
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import Adam

from dataset import get_loader
from model import ImageCaptioningModel
from vocabulary import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Train Image Captioning Model")

    # Paths
    parser.add_argument(
        "--image-dir",
        type=str,
        default="data/train2014",
        help="Directory with training images",
    )
    parser.add_argument(
        "--caption-file",
        type=str,
        default="data/annotations/captions_train2014.json",
        help="COCO captions JSON file",
    )
    parser.add_argument(
        "--val-image-dir",
        type=str,
        default="data/val2014",
        help="Directory with validation images",
    )
    parser.add_argument(
        "--val-caption-file",
        type=str,
        default="data/annotations/captions_val2014.json",
        help="Validation captions JSON file",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default="outputs/vocab.pkl",
        help="Vocabulary pickle file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/base_model",
        help="Directory to save outputs",
    )

    # Model parameters
    parser.add_argument(
        "--embed-size", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument("--hidden-size", type=int, default=512, help="LSTM hidden size")
    parser.add_argument(
        "--num-layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.5, help="Dropout probability"
    )

    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loader workers"
    )

    # Fine-tuning
    parser.add_argument(
        "--fine-tune-epoch",
        type=int,
        default=5,
        help="Epoch to start fine-tuning CNN (0 to disable)",
    )

    # Save/Load
    parser.add_argument(
        "--save-every", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    return parser.parse_args()


def train_epoch(model, data_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()

    total_loss = 0.0
    num_batches = len(data_loader)

    for i, (images, captions, lengths) in enumerate(data_loader):
        images = images.to(device)
        captions = captions.to(device)

        # Forward pass
        outputs = model(images, captions, lengths)

        # Compute loss (exclude <start> token from targets)
        targets = captions[:, 1:]

        # Pack targets for loss computation
        targets = nn.utils.rnn.pack_padded_sequence(
            targets, lengths.cpu() - 1, batch_first=True, enforce_sorted=True
        )[0]

        outputs = nn.utils.rnn.pack_padded_sequence(
            outputs, lengths.cpu() - 1, batch_first=True, enforce_sorted=True
        )[0]

        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

        # Print progress
        if (i + 1) % 100 == 0:
            avg_loss = total_loss / (i + 1)
            perplexity = math.exp(avg_loss)
            print(
                f"  Batch [{i + 1}/{num_batches}] | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f}"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, data_loader, criterion, device):
    """Validate model"""
    model.eval()

    total_loss = 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for images, captions, lengths in data_loader:
            images = images.to(device)
            captions = captions.to(device)

            # Forward pass
            outputs = model(images, captions, lengths)

            # Compute loss
            targets = captions[:, 1:]

            targets = nn.utils.rnn.pack_padded_sequence(
                targets, lengths.cpu() - 1, batch_first=True, enforce_sorted=True
            )[0]

            outputs = nn.utils.rnn.pack_padded_sequence(
                outputs, lengths.cpu() - 1, batch_first=True, enforce_sorted=True
            )[0]

            loss = criterion(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def plot_metrics(history, output_dir):
    """Plot training and validation metrics"""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history["train_loss"], "r-", label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], "b-", label="Val Loss", linewidth=2)
    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_plot.png"), dpi=150)
    plt.close()

    # Perplexity plot
    train_pp = [math.exp(loss) for loss in history["train_loss"]]
    val_pp = [math.exp(loss) for loss in history["val_loss"]]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_pp, "r-", label="Train Perplexity", linewidth=2)
    plt.plot(epochs, val_pp, "b-", label="Val Perplexity", linewidth=2)
    plt.title("Training and Validation Perplexity", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perplexity_plot.png"), dpi=150)
    plt.close()

    print(f"[INFO] Plots saved to {output_dir}")


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load vocabulary
    vocab = Vocabulary.load(args.vocab_file)
    print(f"[INFO] Vocabulary size: {len(vocab)}")

    # Image transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Data loaders
    print("[INFO] Creating data loaders...")
    train_loader = get_loader(
        args.image_dir,
        args.caption_file,
        vocab,
        train_transform,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = get_loader(
        args.val_image_dir,
        args.val_caption_file,
        vocab,
        val_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"[INFO] Train batches: {len(train_loader)}")
    print(f"[INFO] Val batches: {len(val_loader)}")

    # Model
    print("[INFO] Initializing model...")
    model = ImageCaptioningModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        num_layers=args.num_layers,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters: {total_params / 1e6:.2f}M")
    print(f"[INFO] Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
    optimizer = Adam(model.parameters(), lr=args.lr)

    # Resume from checkpoint
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_pp": [], "val_pp": []}

    if args.resume:
        print(f"[INFO] Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]

    # Training loop
    print(f"\n{'=' * 60}")
    print("Starting Training...")
    print(f"{'=' * 60}\n")

    best_val_loss = float("inf")
    total_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()

        print(f"\n[Epoch {epoch + 1}/{args.epochs}]")

        # Fine-tune CNN if specified
        if args.fine_tune_epoch > 0 and epoch == args.fine_tune_epoch:
            model.encoder.unfreeze()
            optimizer = Adam(model.parameters(), lr=args.lr * 0.1)

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        train_pp = math.exp(train_loss)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        val_pp = math.exp(val_loss)

        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_pp"].append(train_pp)
        history["val_pp"].append(val_pp)

        epoch_time = time.perf_counter() - epoch_start

        print(f"\n[Results] Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train PP: {train_pp:.4f}")
        print(f"  Val Loss: {val_loss:.4f} | Val PP: {val_pp:.4f}")
        print(f"  Time: {timedelta(seconds=int(epoch_time))}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "history": history,
                    "args": args,
                },
                checkpoint_path,
            )
            print(f"  Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved (Val Loss: {val_loss:.4f})")

    total_time = time.perf_counter() - total_start
    avg_epoch_time = total_time / args.epochs

    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"  Total Time: {timedelta(seconds=int(total_time))}")
    print(f"  Avg Epoch Time: {timedelta(seconds=int(avg_epoch_time))}")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Best Val PP: {math.exp(best_val_loss):.4f}")
    print(f"{'=' * 60}\n")

    # Plot metrics
    plot_metrics(history, args.output_dir)

    # Save final checkpoint
    final_checkpoint = os.path.join(args.output_dir, "final_checkpoint.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "args": args,
        },
        final_checkpoint,
    )
    print(f"[INFO] Final checkpoint saved: {final_checkpoint}")


if __name__ == "__main__":
    main()
