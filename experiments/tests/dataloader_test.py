import AirysLoader as rsLoad
import torch

NUM_WORKERS = 0
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple Silicon Metal Performance Shaders
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
vocab_size = 50257
output_dim = 256
context_length = 1024

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

position_embedding_layer = torch.nn.Embedding(context_length, output_dim)

batch_size = 8
max_length = 4

dataloader = rsLoad.create_dataloader(
    raw_text,
    batch_size = batch_size,
    max_length = max_length,
    stride = max_length,
    num_workers=NUM_WORKERS
)


for batch in dataloader:
    x,y = batch
    token_embeddings = token_embedding_layer(x)
    pos_embeddings = position_embedding_layer(torch.arange(max_length))

    input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
print(f"Here's a snippet of the embeddings: \n {input_embeddings[:, :, :5]}")