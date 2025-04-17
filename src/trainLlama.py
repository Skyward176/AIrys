import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM, Conv1D
from airysLib.config import ConfigLlama
from safetensors.torch import load_file
from airysLlama.LlamaGenerate import generate, text_to_token_ids, token_ids_to_text
from airysLlama.loadLlamaFromPretrained import loadLlamaFromPretrained
from datasets import load_dataset

from transformers import Trainer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training

def main():
    if torch.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    repo_id = "meta-llama/Llama-3.2-3B-Instruct"
    #repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=nf4_config,
    )
    model = prepare_model_for_kbit_training(model)
    def get_specific_layer_names(model):
        # Create a list to store the layer names
        layer_names = []
        
        # Recursively visit all modules and submodules
        for name, module in model.named_modules():
            # Check if the module is an instance of the specified layers
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv2d, Conv1D)):
                # model name parsing 

                layer_names.append('.'.join(name.split('.')[4:]).split('.')[0])
        
        return layer_names
    loraConfig = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=get_specific_layer_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, loraConfig)

    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    dataset = load_dataset("csv", data_files="training_data/fine/airys_character_dataset.csv", split="train").shuffle()

    def apply_chat_template(input):
        messages = [
            {"role": "user", "content": input['question']},
            {"role": "assistant", "content": input['answer']}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {"prompt": prompt}
    dataset = dataset.map(apply_chat_template)
    dataset = dataset.train_test_split(0.05)
    def tokenize_function(input):
        tokens = tokenizer(input['prompt'], padding="max_length", truncation=True, max_length=128)
        # Set padding token labels to -100 to ignore them in loss calculation
        tokens['labels'] = [
            -100 if token == tokenizer.pad_token_id else token for token in tokens['input_ids']
        ]
        return tokens
    tokenized_dataset = dataset.map(tokenize_function)
    # tokenized_dataset = tokenized_dataset.remove_columns(['question', 'answer', 'prompt'])

    model.train()
    training_args = TrainingArguments(
        output_dir="models/airysLlama/results",
        eval_strategy="steps",  # to evaluate during training
        eval_steps=200,
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=2,
        auto_find_batch_size=True,
        num_train_epochs=40,  # How many times to loop through the dataset
        bf16=True,
        report_to="none", # Here we can use something like tensorboard to see the training metrics
        log_level="info",
        learning_rate=2e-4, # Would avoid larger values here
        max_grad_norm=2, # Clipping the gradients is always a good idea
    )

    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset = tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer = tokenizer 
    )

    trainer.train()

    trainer.save_model("models/airysLlama/airys_llama_character_3B")

if __name__ == "__main__":
    main()