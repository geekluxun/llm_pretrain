import os
import urllib.request

import matplotlib.pyplot as plt
import tiktoken
import torch

from model.llama3_model import Llama3Model
from train.pre_train import plot_losses
from train.pre_train import train_model_simple
from utils.dataprocess import create_dataloader_for_llama3
from utils.device import get_device

temp_path = ".temp/"


def main(gpt_config, settings):
    torch.manual_seed(123)
    device = get_device()

    ##############################
    # Download data if necessary
    ##############################

    file_path = temp_path + "/the-verdict.txt"

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = Llama3Model(gpt_config)
    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_for_llama3(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_for_llama3(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    LLAMA32_CONFIG = {
        "vocab_size": 128_256,  # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 3072,  # Embedding dimension
        "n_heads": 24,  # Number of attention heads
        "n_layers": 28,  # Number of layers
        "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,  # The base in RoPE's "theta"
        "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        "rope_freq": {  # RoPE frequency scaling
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }

    OTHER_SETTINGS = {
        # "learning_rate": 5e-4,
        "learning_rate": 1e-5,
        "num_epochs": 5,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(LLAMA32_CONFIG, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig(temp_path + "/llama3_loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), temp_path + "llama3_model.pth")
    model = Llama3Model(LLAMA32_CONFIG)
    model.load_state_dict(torch.load(temp_path + "llama3_model.pth", weights_only=True))
