def apply_chat_template(input, tokenizer):
    messages = [
        {"role": "user", "content": input['question']},
        {"role": "assistant", "content": input['answer']}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return {"prompt": prompt}