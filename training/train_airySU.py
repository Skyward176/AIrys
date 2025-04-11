import random
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from transformers import AutoTokenizer

from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from airysModels.airySU import airySU

from airysLib.AirysLoader import CausalDataset
from airySU.generateSU import generate
from airysLib.Config import ConfigSU
from airysLib.collate_input import random_causal_collate

from airysLib.tokenIO import text_to_token_ids, token_ids_to_text
import time

def calc_loss_batch(model, input_batch, target_batch, device): # evaluate the loss on one forward pass of a given batch size
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass
    logits = model(input_batch)
    
    # Compute loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                           target_batch.view(-1), ignore_index=-100)  # ignore index for padding
    
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

def evaluate_model(model, train_loader, val_loader, device, eval_iter, dtype=torch.float32): # evaluate the model on the training and validation set
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
            top_k=25,
            eos_id=50256
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
        for input_ids, label_ids in train_loader:
            elapsed_time = []
            start_time = time.time()
            optimizer.zero_grad()
            global_step +=1 
            loss = calc_loss_batch(
                model, input_ids, label_ids, device
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            lr_scheduler.step()
            if global_step % 10 == 0:
                percent_tokens_seen = (global_step / len(train_loader)) * 100
                print(f"Completion:{percent_tokens_seen:.2f}%")
            if global_step % eval_freq == 0:
                print(f"Elapsed time for training this batch: {time.time() - start_time:.2f} seconds")
                start_time = time.time()
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                print(f"Elapsed time for evaluating this batch: {time.time() - start_time:.2f} seconds")
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
                generate_and_print_sample(
                    model, config.context_length, tokenizer, device, start_context
                )

    return train_losses, val_losses, track_tokens_seen

def main(config, settings):

    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon Metal Performance Shaders
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    dataset= load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    dataset = dataset.shard(num_shards=1000, index = 1) # .1% of all wikipedia (lol)
    dataset= dataset.train_test_split(test_size=0.1)  # 10% for validation
    train = dataset["train"]
    val = dataset["test"]
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(train):
        return tokenizer(train["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=config.context_length)
    tokenized_train = train.map(tokenize_function, batched=True, remove_columns=["text"], num_proc = 12)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask"])
    tokenized_val = val.map(tokenize_function, batched=True, remove_columns=["text"], num_proc = 12)
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_dataset = CausalDataset(tokenized_train)
    val_dataset = CausalDataset(tokenized_val)

    def random_causal_collate(batch, max_length=config.context_length, min_length=8):
        input_seqs = []
        label_seqs = []

        for item in batch:
            input_ids = item[0]  # input_ids from the dataset
            # Pick a random crop length
            seq_len = random.randint(min_length, min(len(input_ids) - 1, max_length))

            # Random starting point for the crop
            start_idx = random.randint(0, len(input_ids) - seq_len - 1)
            cropped_input = input_ids[start_idx:start_idx + seq_len]
            cropped_label = input_ids[start_idx + 1:start_idx + seq_len + 1]

            input_seqs.append(torch.tensor(cropped_input, dtype=torch.long))
            label_seqs.append(torch.tensor(cropped_label, dtype=torch.long))

        # Pad all sequences in the batch
        input_batch = pad_sequence(input_seqs, batch_first=True, padding_value=tokenizer.encode(tokenizer.pad_token)[0])
        label_batch = pad_sequence(label_seqs, batch_first=True, padding_value=-100)  # ignore index

        return input_batch, label_batch

    trainLoader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=random_causal_collate)
    valLoader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=random_causal_collate)

    model_path = Path(".")/"models/airySU.pth"
    # Plot results
    if not model_path.exists():
        print(f"Could not find the {model_path} file. Please run the chapter 5 code (ch05.ipynb) to generate the model.pth file.")

    checkpoint = torch.load(model_path, weights_only=True)
    model = airySU(config)
    model.load_state_dict(checkpoint)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["max_lr"], weight_decay=settings["weight_decay"])

    train_losses, val_losses, tokens_seen = trainAirys(
        model, config, trainLoader, valLoader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=200, eval_iter=1,
        start_context="Hi, nice to meet you Irys. You are my daughter. How do you feel?",
        tokenizer=tokenizer,
        max_lr = settings["max_lr"],
        warmup_proportion=settings["warmup_proportion"],
        max_norm = settings["max_norm"]
    )

    model_path = Path(".")/"models/airySU.pth"
    epochs_tensor = torch.linspace(0, TRAINING_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), model_path)
    model = airySU(config)

def print_model_parameters(model):
    """
    Prints the total number of parameters and the number of trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

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
    plt.show()

if __name__ == "__main__":
    config = ConfigSU(
        vocab_size = 50257, #default, overridden later
        emb_dim = 1024,
        n_layers = 32,
        n_heads = 20,
        latent_qkv_dim = 128,
        d_rope = 16,
        context_length = 2048,
        batch_size = 2,
        hidden_dim = 2048,
        dtype= torch.bfloat16,  # Use float16 for training
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
