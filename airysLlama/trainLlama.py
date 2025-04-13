import torch
from huggingface_hub import hf_hub_download
from airysLlama.airysLlama import airysLlama
from airysLlama.LlamaTokenizer import Tokenizer, ChatFormat
from airysLib.config import ConfigLlama
from safetensors.torch import load_file
from airysLlama.LlamaGenerate import generate, text_to_token_ids, token_ids_to_text
from airysLlama.loadLlamaFromPretrained import loadLlamaFromPretrained
from airysLlama.apply_llama_chat_template import apply_chat_template
from datasets import load_dataset

from transformers import Trainer, TrainingArguments
LLAMA_SIZE_STR = "1B"


def test_llama(model, tokenizer, chat_tokenizer, config, device, prompt="Airys,What do llamas eat?"):

    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(prompt, chat_tokenizer).to(device),
        max_new_tokens=1000,
        context_size=config.context_length,
        top_k=1,
        temperature=0.,
        eos_id=chat_tokenizer.tokenizer.special_tokens["<|eot_id|>"]
    )

    output_text = token_ids_to_text(token_ids, tokenizer)
    print(output_text)

    def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
        # Find the index of the first occurrence of "<|end_header_id|>"
        index = text.find(header_end)

        if index != -1:
            # Return the substring starting after "<|end_header_id|>"
            return text[index + len(header_end):].strip()  # Strip removes leading/trailing whitespace
        else:
            # If the token is not found, return the original text
            return text

    print("Output text:\n", clean_text(output_text))
def main():
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    model, tokenizer, chat_tokenizer, config = loadLlamaFromPretrained("1B") 
    model.to(device)
    dataset = load_dataset("csv", data_files="training_data/fine/frijoles_100_conversational.csv", split="train")
    dataset = dataset.map(apply_chat_template)
    def tokenize_function(input):
        tokens = tokenizer(input['prompt'], padding="max_length", truncation=True, max_length=128)
        # Set padding token labels to -100 to ignore them in loss calculation
        tokens['labels'] = [
            -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
        ]
        return tokens
    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'prompt'])

    model.train()
    training_args = TrainingArguments(
        output_dir="models/airysLlama/results",
        eval_strategy="steps",  # to evaluate during training
        eval_steps=40,
        logging_steps=40,
        save_steps=150,
        per_device_train_batch_size=2,  # Adjust based on your hardware
        per_device_eval_batch_size=2,
        num_train_epochs=2,  # How many times to loop through the dataset
        fp16=False,  # Must be False for MacBooks
        report_to="none", # Here we can use something like tensorboard to see the training metrics
        log_level="info",
        learning_rate=1e-5, # Would avoid larger values here
        max_grad_norm=2 # Clipping the gradients is always a good idea
    )

    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer = tokenizer 
    )

    trainer.train()

    trainer.save_model("models/airysLlama/airys_llama_character")
    trainer.save_pretrained("models/airysLlama/airys_llama_character")

if __name__ == "__main__":
    main()