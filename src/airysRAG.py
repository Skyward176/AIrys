# My framework for AIrys' memory system will go here.

# I want to use a two model system, with one model handling the retrieval and storage, while another handles the generation.
# This feels organic and fun to me, like parts of your brain talking across synapses.


from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3n-e2b"