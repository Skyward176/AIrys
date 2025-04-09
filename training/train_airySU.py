
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from transformers import T5Tokenizer
import tiktoken
import math
from pathlib import Path
from airysModels.airySU import airySU

from airysLib.AirysLoader import create_dataloader as rsLoader
from airysApps.AirysGen import generate
from airysLib.Config import ConfigSU

from airysLib.tokenIO import text_to_token_ids, token_ids_to_text


def train_test(epochs,config):
    model = airySU(config).cuda()
    print_model_parameters(model)
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
                               (config.batch_size, config.context_length + 1)).cuda()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits =  model(inputs[:, :-1])
            loss = F.cross_entropy(logits.view(-1, config.vocab_size),
                                   inputs[:, 1:].contiguous().view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        lr_scheduler.step()

        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
def calc_loss_batch(model, input_batch, target_batch, device): # evaluate the loss on one forward pass of a given batch size
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass
    logits = model(input_batch)
    
    # Compute loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           target_batch.view(-1))
    
    return loss

def calc_loss_loader(model, loader, device, num_batches = None): # evaluate the loss of the model with respect to all batches in a dataloader
    total_loss = 0.0
    if len(loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(loader)
    else:
        num_batches = min(num_batches, len(loader))
    for i, (input_batch, target_batch) in enumerate(loader):
        if i < num_batches:
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval() # eval mode (no gradient)

    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(model, val_loader, device, num_batches=eval_iter)

    model.train() # back to train mode

    return train_loss, val_loss
def generate_and_print_sample(model, context_length, tokenizer, device, start_context):
    model.eval()
    context_size = context_length
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate(
            model=model, idx=encoded,
            max_new_tokens=50,
            context_size=context_size,
            temperature=1.5,
            top_k=25
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
def trainAirys(model, config, train_loader, val_loader, optimizer, device,
                num_epochs, eval_freq, eval_iter, start_context,
                tokenizer, max_lr, warmup_proportion, max_norm):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,  # Increased max LR
        total_steps=num_epochs * len(train_loader),
        pct_start=warmup_proportion,  # Shorter warmup
        anneal_strategy='cos'
    )

    global_step = -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            global_step +=1 

            loss = calc_loss_batch(
                model, input_batch, target_batch, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            lr_scheduler.step()
            tokens_seen += input_batch.numel()  # Count tokens seen

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                total_tokens = len(train_loader) * config.batch_size * config.context_length
                percent_tokens_seen = (tokens_seen / total_tokens) * 100
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                print(f"Tokens seen: {tokens_seen} ({percent_tokens_seen:.2f}%)")
                # generate a sample
                # generate_and_print_sample(
                #     model, config.context_length, tokenizer, device, start_context
                # )
    return train_losses, val_losses, track_tokens_seen

def main(config, settings):

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon Metal Performance Shaders
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    tokenizer = tiktoken.get_encoding("gpt2")
    config.vocab_size = tokenizer.n_vocab # set vocab size to the tokenizer vocab size
    file_path = "training_data.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    
    model = airySU(config)

    model_path = Path(".")/"models/airySU.pth"

    # model loading for later
    # try:
    #     model.load_state_dict(torch.load(model_path, weights_only=True))
    # except FileNotFoundError:
    #     print("Model weights not found. Starting training from scratch.")

    # train_test(100,config)
    print_model_parameters(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["max_lr"], weight_decay=settings["weight_decay"])


    train_ratio =0.9
    split_idx = int(len(text_data) * train_ratio)

    train_loader = rsLoader(
        text_data[:split_idx],
        batch_size=config.batch_size,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )
    val_loader = rsLoader(
        text_data[split_idx:],
        batch_size=config.batch_size,
        max_length=config.context_length,
        stride=config.context_length,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )


    train_losses, val_losses, tokens_seen = trainAirys(
        model, config, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Hi, nice to meet you Irys. You are my daughter. How do you feel?",tokenizer=tokenizer,
        max_lr = settings["max_lr"],
        warmup_proportion=settings["warmup_proportion"],
        max_norm = settings["max_norm"]
    )
def print_model_parameters(model):
    """
    Prints the total number of parameters and the number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    config = ConfigSU(
        vocab_size = 32000, #default, overridden later
        emb_dim = 1600,
        n_layers = 48,
        n_heads = 24,
        latent_qkv_dim = 128,
        d_rope = 16,
        context_length = 1024,
        batch_size = 1,
        drop_rate = 0.0,
    )

    TRAINING_SETTINGS = {
        "max_lr": 3e-4,
        "num_epochs": 1,
        "weight_decay": 0.1,
        "warmup_proportion": 0.05,
        "max_norm": 1.0,
        "d_type": torch.bfloat16,  # Use float16 for training
    }

    ###########################
    # Initiate training
    ###########################

    main(config, TRAINING_SETTINGS)
