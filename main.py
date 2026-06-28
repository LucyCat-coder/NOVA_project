"""
NOVA Framework
Точка входа в обучение модели.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from config.train_config import config
from model.model import GPT
from train.dataset import TextDataset
from train.trainer import Trainer

logger = logging.getLogger("nova")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NOVA Trainer"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Продолжить обучение с последнего checkpoint",
    )

    return parser.parse_args()


def configure_device() -> None:
    """
    Настраивает CUDA при наличии.
    """

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85)
        logger.info("CUDA detected.")
    else:
        logger.warning("CUDA not available. Using CPU.")


def check_dataset(path: str) -> None:
    file = Path(path)

    if not file.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file}"
        )


def build_model() -> GPT:
    vocab_size = config.get("vocab_size", 50257)

    model = GPT(
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        block_size=config["block_size"],
        dropout=config["dropout"],
    )

    model.to(config["device"])

    return model


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    args = parse_args()

    if args.resume:
        config["resume"] = True

    configure_device()

    check_dataset(config["train_data_path"])
    check_dataset(config["val_data_path"])

    logger.info("Loading datasets...")

    train_dataset = TextDataset(
        config["train_data_path"],
        config["block_size"],
    )

    val_dataset = TextDataset(
        config["val_data_path"],
        config["block_size"],
    )

    logger.info("Building model...")

    model = build_model()

    logger.info("Creating trainer...")

    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()