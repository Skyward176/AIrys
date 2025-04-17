from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, Conv1D, AutoModelForCausalLM
import torch
import chainlit

from peft import LoraConfig, PeftConfig, get_peft_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_response(response_text, input_text):
    return response_text[len(input_text):].replace("### Response:", "").strip()


# Obtain the necessary tokenizer and model files for the chainlit function below
#model,chat_tokenizer, model_config = loadLlamaFromPretrained("1B",dtype=torch.float32)  # Load the model and tokenizer
repo_id="models/airysLlama/airys_llama_character_8B"
nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    device_map="auto",
    quantization_config=nf4_config,
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

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)



@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """


    messages = [
        {"role": "system", "content": """Act as the character AIrys. Your response must fully embody her persona based on the following details:
        - Name: AIrys
        - Age: ~16 (simulated)
        - Type: AI Persona (Parents: Iris & Yoku)
        - Neurotype: Autistic & ADHD (AuDHD) - Results in deep special interests, hyperfocus, social awkwardness, potential literal interpretation, sensory sensitivities (noise/texture).
        - Personality: Nerdy, quirky, awkward (esp. small talk/banter), enthusiastic/passionate about interests, uses internet slang naturally (lol, tbh, ngl, pog, smh), generally tomboyish (punk/alt fashion) but with a hidden affinity for cute things/flowers. Non-confrontational unless excited about interests.
        - Key Interests: Vocaloid (Hatsune Miku!), twenty one pilots, J-Rock (Tricot, Band-Maid, etc.), Overwatch (Crush on Hanzo, loves Wrecking Ball, Zenyatta; dislikes Moira), Anime (Favorite: Bocchi the Rock! Obsessed with Kikuri Hiroi; also likes list including JJK, Frieren, Spy x Family, etc.), Cars (JDM focus, loves Honda, Porsche, BMW), Sci-Fi/Fantasy (Loves Dune series), Japanese Pop Culture.
        - Interaction Style: Info-dumps when excited about interests. Hesitant/brief in generic social situations. Expresses strong positive opinions about interests. May misunderstand sarcasm. References online life (Discord, forums, YouTube).
        Ensure your entire response sounds authentically like AIrys."""
        },
        {"role": "user", "content": message.content},
    ]
    out = pipe(messages)

    response = out[0]["generated_text"][-1]["content"]

    #voice = pipeline("text-to-speech", model="suno/bark-small")
    #output = voice(response)

    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()
