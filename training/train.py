import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from transformers import T5Tokenizer
import tiktoken
import math
from pathlib import Path
from airysModels.airysDeep import airysDeep

from airysLib.AirysLoader import create_dataloader as rsLoader
from airysApps.AirysGen import generate
from airysLib.config import Config

from airysLib.tokenIO import text_to_token_ids, token_ids_to_text

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits, aux_loss = model(input_batch)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
    loss += 0.0001 * aux_loss  # Auxiliary loss for MoE
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
    context_size = 1024
    #context_size = model.pos_emb.weight.shape[0]
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


def train_test(epochs,config):
    model = airysDeep(config).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Learning rate schedule with warmup
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,  # Increased max LR
        total_steps=epochs,
        pct_start=0.1,  # Shorter warmup
    )

    for epoch in range(epochs):
        # Use structured inputs
        inputs = torch.randint(0, config.vocab_size // 10,
                               (config.batch_size, config.seq_len + 1)).cuda()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits, aux_loss = model(inputs[:, :-1])
            loss = F.cross_entropy(logits.view(-1, config.vocab_size),
                                   inputs[:, 1:].contiguous().view(-1))
            loss += 0.0001 * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")

def trainAirys(model,config, train_loader, val_loader, optimizer, device, num_epochs,
                    eval_freq, eval_iter, start_context, tokenizer,
                       init_learn_rate, min_learn_rate, warmup_proportion, max_norm):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0

    peak_learn_rate = optimizer.param_groups[0]["lr"]
    min_learn_rate = 0.1 * init_learn_rate
    track_lrs = []

    total_steps = len(train_loader) * num_epochs
    warmup_steps = warmup_proportion * total_steps
    lr_increment = (peak_learn_rate - init_learn_rate) / warmup_steps
    # Main training loop


    global_step = -1
    for param in model.parameters():
        param.requires_grad = True

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        print(model)
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration

            global_step += 1

            if global_step < warmup_steps:
                lr = init_learn_rate + global_step * lr_increment
            else:
                progress = ((global_step - warmup_steps) /
                            (total_steps - warmup_steps))
                lr = min_learn_rate + (peak_learn_rate - min_learn_rate) * 0.5 * (
                    1 + math.cos(math.pi * progress))

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

                #generate_and_print_sample(
                #    model, tokenizer, device, start_context
                #)
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


def main(config, settings):

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
    # train_test(100,config)
    model = airysDeep(config)
    print_model_parameters(model)
    model_path = Path(".")/"model.pth"
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
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
        max_length=config.seq_len,
        stride=config.seq_len,
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    val_loader = rsLoader(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=config.seq_len,
        stride=config.seq_len,
        drop_last=False,
        shuffle=False,
        num_workers=0
    )


    tokenizer = T5Tokenizer.from_pretrained("t5-small") 

    train_losses, val_losses, tokens_seen = trainAirys(
        model, config, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=1, eval_iter=1,
        start_context="Hi my name is AIrys",tokenizer=tokenizer,
        min_learn_rate=settings["min_learn_rate"],
        init_learn_rate=settings["init_learn_rate"],
        warmup_proportion=settings["warmup_proportion"],
        max_norm = settings["max_norm"]
    )

    return train_losses, val_losses, tokens_seen, model

def print_model_parameters(model):
    """
    Prints the total number of parameters and the number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    config = Config(
        vocab_size = 32000,
        d_in = 5120,
        d_out = 5120,
        n_layers = 4,
        n_heads = 8,
        d_kv_comp = 128,
        d_rope = 16,
        n_experts = 32,
        n_shared = 2,
        top_k = 2,
        seq_len = 256,
        batch_size = 12,
        ffn_dim = 384,
        device_groups = 1
    )

    OTHER_SETTINGS = {
        "peak_learn_rate": 3e-4,
        "init_learn_rate": 3e-5,  # Typically 10x smaller than peak learn rate
        "min_learn_rate": 3e-6,  # Typically 10x smaller than init learn rate
        "num_epochs": 1,
        "batch_size": 12,
        "weight_decay": 0.1,
        "warmup_proportion": 0.05,
        "max_norm": 1.0,
        "d_type": torch.bfloat16,  # Use float16 for training
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(config, OTHER_SETTINGS)

    ###########################
    # After training
    ###########################

    # Plot results
    model_path = Path(".")/"model.pth"
    # epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    # plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    # plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), model_path)
    model = airysDeep(config)
    model.load_state_dict(torch.load(model_path, weights_only=True))