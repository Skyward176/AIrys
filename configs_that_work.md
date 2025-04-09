# Configs that work on my setup

Yes, I know there are more effective ways to do this but *for now* this will do

### GPT Medium

Batch size of 6, standard settings otherwise

### Mix of Experts with Latent attention (deepseek clone)

32k vocab
huggingface tokenizer

keeping feed forward layer depth proportional to model input/output width

keep the context length low

context length can't be much larger than ff width or dim in or out

 config = Config(
        vocab_size = 32000,
        d_in = 7680,
        d_out = 7680,
        n_layers = 4,
        n_heads = 12,
        d_kv_comp = 128,
        d_rope = 32,
        n_experts = 32,
        n_shared = 2,
        top_k = 2,
        seq_len = 768,
        batch_size = 1,
        ffn_dim = 768,
        device_groups = 1
    )