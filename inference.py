"""
Inference Script for Image Captioning
Generate captions for external images
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from matplotlib.gridspec import GridSpec
from PIL import Image

from model import ImageCaptioningModel
from vocabulary import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Generate captions for images")

    parser.add_argument(
        "--image-dir",
        type=str,
        default="val_images",
        help="Directory with images to caption",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model checkpoint"
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
        default="inference_results",
        help="Directory to save results",
    )

    # Model parameters (should match training)
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=1)

    # Inference parameters
    parser.add_argument(
        "--beam-search", action="store_true", help="Use beam search instead of greedy"
    )
    parser.add_argument(
        "--beam-width", type=int, default=3, help="Beam width for beam search"
    )
    parser.add_argument(
        "--max-length", type=int, default=20, help="Maximum caption length"
    )

    return parser.parse_args()


def load_image(image_path, transform):
    """Load and transform image"""
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor


def caption_to_text(caption_indices, vocab):
    """Convert caption indices to text"""
    words = []
    for idx in caption_indices:
        word = vocab.idx2word[idx]
        if word == "<end>":
            break
        if word not in ["<start>", "<pad>"]:
            words.append(word)
    return " ".join(words)


def generate_captions(
    model,
    image_dir,
    vocab,
    transform,
    device,
    beam_search=False,
    beam_width=3,
    max_length=20,
):
    """Generate captions for all images in directory"""
    model.eval()

    results = []

    # Get all image files
    image_files = [
        f
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    print(f"[INFO] Found {len(image_files)} images")

    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(image_dir, img_file)

            try:
                # Load image
                image, image_tensor = load_image(img_path, transform)
                image_tensor = image_tensor.to(device)

                # Generate caption
                caption_indices = model.generate_caption(
                    image_tensor,
                    beam_search=beam_search,
                    beam_width=beam_width,
                    max_length=max_length,
                )

                # Convert to text
                caption_text = caption_to_text(caption_indices, vocab)

                results.append(
                    {"filename": img_file, "image": image, "caption": caption_text}
                )

                print(f"[{img_file}] {caption_text}")

            except Exception as e:
                print(f"[ERROR] Failed to process {img_file}: {e}")

    return results


def visualize_results(results, output_dir, model_name="Model"):
    """Create visualization of results"""
    os.makedirs(output_dir, exist_ok=True)

    num_images = len(results)
    cols = 3
    rows = (num_images + cols - 1) // cols

    fig = plt.figure(figsize=(15, 5 * rows))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.4, wspace=0.3)

    for idx, result in enumerate(results):
        row = idx // cols
        col = idx % cols

        ax = fig.add_subplot(gs[row, col])
        ax.imshow(result["image"])
        ax.set_title(
            f"{result['filename']}\n{result['caption']}", fontsize=10, wrap=True
        )
        ax.axis("off")

    plt.suptitle(f"Image Captioning Results - {model_name}", fontsize=16, y=0.995)
    plt.tight_layout()

    output_path = os.path.join(
        output_dir, f"results_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[INFO] Visualization saved to {output_path}")


def save_results_txt(results, output_dir, model_name="Model"):
    """Save results to text file"""
    output_path = os.path.join(
        output_dir, f"captions_{model_name.lower().replace(' ', '_')}.txt"
    )

    with open(output_path, "w") as f:
        f.write(f"Image Captioning Results - {model_name}\n")
        f.write("=" * 60 + "\n\n")

        for result in results:
            f.write(f"Image: {result['filename']}\n")
            f.write(f"Caption: {result['caption']}\n\n")

    print(f"[INFO] Captions saved to {output_path}")


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

    # Image transformation (validation mode)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load model
    print(f"[INFO] Loading model from {args.model_path}")
    model = ImageCaptioningModel(
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        vocab_size=len(vocab),
        num_layers=args.num_layers,
    ).to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("[INFO] Model loaded successfully")

    # Determine model name
    search_type = "Beam Search" if args.beam_search else "Greedy"
    model_name = f"{os.path.basename(args.model_path).split('.')[0]} ({search_type})"

    print(f"\n{'=' * 60}")
    print(f"Generating Captions with {search_type}")
    print(f"{'=' * 60}\n")

    # Generate captions
    results = generate_captions(
        model,
        args.image_dir,
        vocab,
        transform,
        device,
        beam_search=args.beam_search,
        beam_width=args.beam_width,
        max_length=args.max_length,
    )

    print(f"\n[INFO] Generated {len(results)} captions")

    # Save results
    visualize_results(results, args.output_dir, model_name)
    save_results_txt(results, args.output_dir, model_name)

    print(f"\n{'=' * 60}")
    print("Inference Complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
