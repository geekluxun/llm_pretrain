import os
import urllib.request

import matplotlib.pyplot as plt
import tiktoken
import torch

from model.llama2_model import Llama2Model
from train.pre_train import plot_losses
from train.pre_train import train_model_simple
from utils.dataprocess import create_dataloader_for_llama2
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

    model = Llama2Model(gpt_config)
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

    train_loader = create_dataloader_for_llama2(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = create_dataloader_for_llama2(
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
    LLAMA2_CONFIG_7B = {
        "vocab_size": 32000,  # Vocabulary size
        "context_length": 256,  # Context length
        "emb_dim": 4096,  # Embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 32,  # Number of layers
        "hidden_dim": 11008,  # NEW: Size of the intermediate dimension in FeedForward
        "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
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

    train_losses, val_losses, tokens_seen, model = main(LLAMA2_CONFIG_7B, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig(temp_path + "/llama2_loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), temp_path + "llama2_model.pth")
    model = Llama2Model(LLAMA2_CONFIG_7B)
    model.load_state_dict(torch.load(temp_path + "llama2_model.pth", weights_only=True))
