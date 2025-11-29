"""
Compare Multiple Model Variants on External Images
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision.transforms as transforms
from matplotlib.gridspec import GridSpec
from PIL import Image

from model import ImageCaptioningModel
from vocabulary import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Compare model variants")

    parser.add_argument(
        "--image-dir",
        type=str,
        default="imagenes_validacion",
        help="Directory with validation images",
    )
    parser.add_argument(
        "--models", nargs="+", required=True, help="List of model checkpoint paths"
    )
    parser.add_argument(
        "--model-names", nargs="+", required=True, help="Names for each model"
    )
    parser.add_argument(
        "--vocab-file", type=str, default="vocab.pkl", help="Vocabulary pickle file"
    )
    parser.add_argument(
        "--output-dir", type=str, default="comparison_results", help="Output directory"
    )

    # Model parameters
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)

    # Inference
    parser.add_argument("--beam-search", action="store_true")
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=20)

    return parser.parse_args()


def load_and_prepare_models(
    model_paths, vocab_size, embed_size, hidden_size, num_layers, device
):
    """Load all model variants"""
    models = []

    for path in model_paths:
        model = ImageCaptioningModel(
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_layers=num_layers,
        ).to(device)

        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
        print(f"[INFO] Loaded model from {path}")

    return models


def caption_to_text(caption_indices, vocab):
    """Convert indices to text"""
    words = []
    for idx in caption_indices:
        word = vocab.idx2word[idx]
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            words.append(word)
    return " ".join(words)


def generate_comparison(
    models,
    model_names,
    image_dir,
    vocab,
    transform,
    device,
    beam_search=False,
    beam_width=3,
    max_length=20,
):
    """Generate captions with all models"""

    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"[INFO] Processing {len(image_files)} images with {len(models)} models")

    results = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)

        try:
            # Load image
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)

            captions = {}

            # Generate with each model
            with torch.no_grad():
                for model, name in zip(models, model_names):
                    caption_indices = model.generate_caption(
                        image_tensor,
                        beam_search=beam_search,
                        beam_width=beam_width,
                        max_length=max_length,
                    )
                    caption_text = caption_to_text(caption_indices, vocab)
                    captions[name] = caption_text

            results.append({"filename": img_file, "image": image, "captions": captions})

            print(f"\n[{img_file}]")
            for name, caption in captions.items():
                print(f"  {name}: {caption}")

        except Exception as e:
            print(f"[ERROR] Failed to process {img_file}: {e}")

    return results


def create_comparison_table(results, model_names, output_dir):
    """Create comparison table"""

    data = []
    for result in results:
        row = {"Image": result["filename"]}
        for name in model_names:
            row[name] = result["captions"][name]
        data.append(row)

    df = pd.DataFrame(data)

    # Save to CSV
    csv_path = os.path.join(output_dir, "comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Comparison table saved to {csv_path}")

    # Save to formatted text
    txt_path = os.path.join(output_dir, "comparison_table.txt")
    with open(txt_path, "w") as f:
        f.write("Model Comparison Results\n")
        f.write("=" * 80 + "\n\n")

        for result in results:
            f.write(f"Image: {result['filename']}\n")
            f.write("-" * 80 + "\n")
            for name in model_names:
                f.write(f"{name:20s}: {result['captions'][name]}\n")
            f.write("\n")

    print(f"[INFO] Formatted table saved to {txt_path}")


def visualize_comparison(results, model_names, output_dir):
    """Create visual comparison"""

    for result in results:
        fig = plt.figure(figsize=(15, 3 + 2 * len(model_names)))
        gs = GridSpec(len(model_names) + 1, 1, figure=fig, hspace=0.4)

        # Show image
        ax_img = fig.add_subplot(gs[0, 0])
        ax_img.imshow(result["image"])
        ax_img.set_title(f"{result['filename']}", fontsize=14, fontweight="bold")
        ax_img.axis("off")

        # Show captions for each model
        for idx, name in enumerate(model_names):
            ax_text = fig.add_subplot(gs[idx + 1, 0])
            caption = result["captions"][name]

            ax_text.text(
                0.5,
                0.5,
                caption,
                ha="center",
                va="center",
                fontsize=12,
                wrap=True,
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
            )
            ax_text.set_title(name, fontsize=12, fontweight="bold")
            ax_text.axis("off")

        plt.tight_layout()

        output_path = os.path.join(output_dir, f"comparison_{result['filename']}")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"[INFO] Individual comparisons saved to {output_dir}")

    # Create grid overview
    num_images = len(results)
    fig = plt.figure(figsize=(20, 5 * num_images))
    gs = GridSpec(num_images, len(model_names) + 1, figure=fig, hspace=0.4, wspace=0.3)

    for img_idx, result in enumerate(results):
        # Image column
        ax = fig.add_subplot(gs[img_idx, 0])
        ax.imshow(result["image"])
        ax.set_title(result["filename"], fontsize=10)
        ax.axis("off")

        # Caption columns
        for model_idx, name in enumerate(model_names):
            ax = fig.add_subplot(gs[img_idx, model_idx + 1])
            caption = result["captions"][name]
            ax.text(0.5, 0.5, caption, ha="center", va="center", fontsize=9, wrap=True)
            if img_idx == 0:
                ax.set_title(name, fontsize=11, fontweight="bold")
            ax.axis("off")

    plt.suptitle("Model Comparison Overview", fontsize=16, y=0.995)
    overview_path = os.path.join(output_dir, "comparison_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Overview saved to {overview_path}")


def main():
    args = parse_args()

    if len(args.models) != len(args.model_names):
        raise ValueError("Number of models must match number of model names")

    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load vocabulary
    vocab = Vocabulary.load(args.vocab_file)
    print(f"[INFO] Vocabulary size: {len(vocab)}")

    # Transform
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load models
    print(f"\n[INFO] Loading {len(args.models)} models...")
    models = load_and_prepare_models(
        args.models,
        len(vocab),
        args.embed_size,
        args.hidden_size,
        args.num_layers,
        device,
    )

    # Generate comparisons
    print(f"\n{'=' * 60}")
    print("Generating Captions for Comparison")
    print(f"{'=' * 60}\n")

    results = generate_comparison(
        models,
        args.model_names,
        args.image_dir,
        vocab,
        transform,
        device,
        beam_search=args.beam_search,
        beam_width=args.beam_width,
        max_length=args.max_length,
    )

    # Create outputs
    print(f"\n[INFO] Creating comparison outputs...")
    create_comparison_table(results, args.model_names, args.output_dir)
    visualize_comparison(results, args.model_names, args.output_dir)

    print(f"\n{'=' * 60}")
    print("Comparison Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
