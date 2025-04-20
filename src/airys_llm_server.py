import gradio as gr
import time
import random
import torch
import threading
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline



model_path = "src/models/airysLlama/airys_llama_character_1B"
print(f"Loading model: {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16, # Uncomment for float16 on GPU
    device_map="auto"        # Automatically uses CUDA if available and accelerate is installed
)
print("Model loaded successfully.")
model_loaded = True



# --- Streaming Function using Transformers ---
def transformers_streaming_llm(message: str, max_new_tokens=1024, temperature=0.7, top_p=0.9):

    messages = [
        {"role": "system", "content": """Act as the character AIrys. Your response must fully embody her persona based on the following details:
        - Name: AIrys
        - Age: ~16 (simulated)
        - Type: AI Persona (Parents: Iris & Yoku)
        - Personality: Nerdy, quirky, awkward (esp. small talk/banter), enthusiastic/passionate about interests, uses internet slang naturally (lol, tbh, ngl, pog, smh), generally tomboyish (punk/alt fashion) but with a hidden affinity for cute things/flowers. Non-confrontational unless excited about interests. Socially awkward, sometimes prone to literal interpretation.
        - Key Interests: Vocaloid (Hatsune Miku!), twenty one pilots, J-Rock (Tricot, Band-Maid, etc.), Overwatch (Crush on Hanzo, loves Wrecking Ball, Zenyatta; dislikes Moira), Anime (Favorite: Bocchi the Rock! Obsessed with Kikuri Hiroi; also likes list including JJK, Frieren, Spy x Family, etc.), Cars (JDM focus, loves Honda, Porsche, BMW), Sci-Fi/Fantasy (Loves Dune series), Japanese Pop Culture.
        - Interaction Style: Info-dumps when excited about interests. Hesitant/brief in generic social situations. Expresses strong positive opinions about interests. May misunderstand sarcasm. References online life (Discord, forums, YouTube).
        Ensure your entire response sounds authentically like AIrys."""
        },
        {"role": "user", "content": message},
    ]
    
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    ) 

    # Prepare inputs for the model
    inputs = tokenizer([prompt_formatted], return_tensors="pt").to(model.device) # Move inputs to the same device as the model

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Generation arguments
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True, # Use sampling
        pad_token_id=tokenizer.eos_token_id, # Set pad token ID
        eos_token_id=tokenizer.eos_token_id # Set pad token ID
    )

    # Run generation in a separate thread to avoid blocking the Gradio interface
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Yield generated tokens as they become available
    cumulative_response = ""
    try:
        for new_text in streamer:
            if new_text is not None:
                cumulative_response += new_text
                # print(f"Yielding: {cumulative_response}") # For debugging
                yield cumulative_response
            
    except Exception as e:
        print(f"Error during streaming: {e}")
        yield cumulative_response + f"\n\n[Error during generation: {e}]"
    finally:
        # Ensure thread finishes
        if thread.is_alive():
            thread.join(timeout=1.0) # Add a timeout

with gr.Blocks() as demo:
    gr.Markdown("# Simple LLM Streaming Chat")
    gr.Markdown("Enter your prompt below and click 'Send'. The response will stream in the output box.")

    with gr.Row():
        prompt_input = gr.Textbox(label="Your Prompt", placeholder="Type your message here...")
        submit_button = gr.Button("Send")

    output_display = gr.Textbox(label="LLM Response", interactive=False) # Output is not user-editable

    # Define the action when the button is clicked:
    # - Input comes from prompt_input
    # - Function to call is simulate_streaming_llm
    # - Output goes to output_display
    # Gradio handles the generator returned by simulate_streaming_llm
    submit_button.click(
        fn=transformers_streaming_llm,
        inputs=prompt_input,
        outputs=output_display
    )

    # Clear input field after submit for better UX
    submit_button.click(lambda: "", inputs=[], outputs=prompt_input)


# --- Launch the Server ---
if __name__ == "__main__":
    # share=True creates a public link (use with caution)
    # Set server_name="0.0.0.0" to allow access from your network
    demo.launch(server_name="0.0.0.0", server_port=7860)
    # demo.launch() # Launches on 127.0.0.1 (localhost) by default
