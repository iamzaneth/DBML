from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "rrrr66254/Glossa-BART"
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models" / "glossa_bart_local"
TRUTHY_VALUES = {"1", "true", "yes", "y"}
REQUIRED_TOKENIZER_FILES = (
    "config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)
MODEL_WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")


def parse_args() -> argparse.Namespace:
    """Read optional inference settings from the command line."""

    parser = argparse.ArgumentParser(
        description="Translate an ASL gloss sequence into an English sentence."
    )
    parser.add_argument(
        "--gloss",
        help="Gloss text to translate once. If omitted, an interactive prompt is used.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Run the model on CPU even when CUDA is available.",
    )
    return parser.parse_args()


def resolve_device(force_cpu: bool = False) -> torch.device:
    """Choose the best available PyTorch device for inference."""

    env_force_cpu = os.getenv("FORCE_CPU", "").strip().lower() in TRUTHY_VALUES
    if force_cpu or env_force_cpu:
        print("Using CPU.")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        print(f"Using CUDA device: {torch.cuda.get_device_name(device_index)}")
        print(f"PyTorch: {torch.__version__}")
        print(f"PyTorch CUDA runtime: {torch.version.cuda}")
        return torch.device(f"cuda:{device_index}")

    print("CUDA is not available. Using CPU.")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    return torch.device("cpu")


def model_files_are_available() -> bool:
    """Return whether all required local Glossa-BART files are present."""

    if not MODEL_DIR.exists():
        return False

    has_tokenizer_files = all(
        (MODEL_DIR / filename).is_file() for filename in REQUIRED_TOKENIZER_FILES
    )
    has_model_weights = any(
        (MODEL_DIR / filename).is_file() for filename in MODEL_WEIGHT_FILES
    )

    return has_tokenizer_files and has_model_weights


def ensure_model_available() -> None:
    """Download Glossa-BART into models/ when local files are missing."""

    if model_files_are_available():
        print(f"Using local model: {MODEL_DIR}")
        return

    print(f"Local model files are missing or incomplete: {MODEL_DIR}")
    print(f"Downloading {MODEL_NAME} to {MODEL_DIR}...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    print(f"Model saved to {MODEL_DIR}")


def load_model(device: torch.device) -> tuple[Any, Any]:
    """Load the tokenizer and sequence-to-sequence model from local files."""

    print("Loading tokenizer from local files...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

    print("Loading model from local files...")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR, local_files_only=True)
    model = model.to(device)
    model.eval()

    print("Model is ready.")
    return tokenizer, model


def normalize_gloss(raw_gloss: str) -> str:
    """Normalize user input into the uppercase token format expected by gloss models."""

    return " ".join(token.strip().upper() for token in raw_gloss.split() if token.strip())


def gloss_to_sentence(
    gloss: str,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    max_length: int = 64,
    num_beams: int = 4,
) -> str:
    """Generate an English sentence from one gloss string."""

    normalized_gloss = normalize_gloss(gloss)
    if not normalized_gloss:
        raise ValueError("Gloss input cannot be empty.")

    inputs = tokenizer(
        normalized_gloss,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def print_translation(gloss: str, tokenizer: Any, model: Any, device: torch.device) -> None:
    """Translate one gloss string and print the result."""

    normalized_gloss = normalize_gloss(gloss)
    sentence = gloss_to_sentence(normalized_gloss, tokenizer, model, device)

    print(f"Gloss: {normalized_gloss}")
    print(f"Sentence: {sentence}")


def run_prompt(tokenizer: Any, model: Any, device: torch.device) -> None:
    """Run an interactive gloss-to-sentence prompt."""

    print("Enter an ASL gloss sequence. Press Enter on an empty line to exit.")

    while True:
        gloss = input("Gloss> ").strip()
        if not gloss:
            break

        try:
            print_translation(gloss, tokenizer, model, device)
        except ValueError as error:
            print(f"Error: {error}")


def main() -> None:
    """Load Glossa-BART and translate user-provided gloss input."""

    args = parse_args()
    device = resolve_device(force_cpu=args.force_cpu)
    ensure_model_available()
    tokenizer, model = load_model(device)

    if args.gloss:
        print_translation(args.gloss, tokenizer, model, device)
        return

    run_prompt(tokenizer, model, device)


if __name__ == "__main__":
    main()
