import matplotlib.pyplot as plt
import torch
import tiktoken
import math

from airysModels.airysGPT2 import airysGPT2

from airysLib.AirysLoader import create_dataloader as rsLoader
from airysApps.AirysGen import generate

from airysLib.tokenIO import text_to_token_ids, token_ids_to_text

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=1.4,
            top_k=50
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def trainAirys(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer,
                       init_learn_rate, min_learn_rate, warmup_proportion, max_norm):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    peak_learn_rate = optimizer.param_groups[0]["lr"]
    min_learn_rate = 0.1 * init_learn_rate
    track_lrs = []

    total_steps = len(train_loader) * num_epochs
    warmup_steps = warmup_proportion * total_steps
    lr_increment = (peak_learn_rate - init_learn_rate) / warmup_steps
    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            global_step += 1

            if global_step < warmup_steps:
                lr = init_learn_rate + global_step*lr_increment
            else:
                progress = ((global_step - warmup_steps)/
                            (total_steps - warmup_steps))
                lr = min_learn_rate + (peak_learn_rate-min_learn_rate) * 0.5 *(
                1+ math.cos(math.pi*progress))
                

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(optimizer.param_groups[0]["lr"])
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients

            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                print(f"current learning rate : {lr}\n")

                generate_and_print_sample(
                    model, tokenizer, device, start_context
                )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon Metal Performance Shaders
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    ##############################
    # Download data if necessary
    ##############################

    file_path = "training_data.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    ##############################
    # Initialize model
    ##############################

    model = airysGPT2(gpt_config)
    try:
        model.load_state_dict(torch.load("../model.pth", weights_only=True))
    except FileNotFoundError:
        print("Model weights not found. Starting training from scratch.")

    model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["peak_learn_rate"], weight_decay=settings["weight_decay"]
    )

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = rsLoader(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = rsLoader(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )


    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = trainAirys(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Hi my name is AIrys",tokenizer=tokenizer,
        min_learn_rate=settings["min_learn_rate"],
        init_learn_rate=settings["init_learn_rate"],
        warmup_proportion=settings["warmup_proportion"],
        max_norm = settings["max_norm"]
    )

    return train_losses, val_losses, tokens_seen, model



if __name__ == "__main__":

    BASE_CONFIG = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1025)
        "drop_rate": 0.0,       # Dropout rate
        "qkv_bias": True       # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-small (124M)"  # Choose the model you want to use
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    print("GPT Config:", BASE_CONFIG)
    print("Model Size:", model_size)

    OTHER_SETTINGS = {
        "peak_learn_rate": 0.001,
        "init_learn_rate": 3e-05,
        "min_learn_rate": 1e-6,
        "num_epochs": 1,
        "batch_size":2,
        "weight_decay": 0.1,
        "warmup_proportion": 0.2,
        "max_norm":1.0
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(BASE_CONFIG, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "../model.pth")
    model = airysGPT2(BASE_CONFIG)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
