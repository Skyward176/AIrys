import torch
from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, Conv1D, AutoModelForCausalLM
from peft import LoraConfig, PeftConfig, get_peft_model

def loadAirys(repo_id="models/airysLlama/airys_llama_character_8B"):
    #nf4_config = BitsAndBytesConfig(
    #    load_in_4bit=True,
    #    bnb_4bit_use_double_quant=True,
    #    bnb_4bit_quant_type="nf4",
    #    bnb_4bit_compute_dtype=torch.bfloat16,
    #)
    #it's way faster to inference non-quantized, apparently
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        device_map="auto",
        use_flash_attention_2=True,
        torch_dtype=torch.bfloat16,
    )
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
    #loraConfig = LoraConfig(
    #    r=16,
    #    lora_alpha=32,
    #    target_modules=get_specific_layer_names(model),
    #    lora_dropout=0.05,
    #    bias="none",
    #    task_type="CAUSAL_LM",
    #)
    #model = get_peft_model(model, loraConfig)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer