from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class AIrysRAG:
    def __init__(self, think_model, remember_model):
        self.thinker = pipeline("text-generation", model=think_model)
        self.rememberer = pipeline("text-generation", model=remember_model)
        self.tokenizer = AutoTokenizer.from_pretrained(think_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, input_text :str):
        out = self.thinker(input_text)
        print(out)
        # memory_decision = self.rememberer("Decide if any of this information is relevant and worth remembering in future conversations with the user: " + input_text + " " + out[0]['generated_text'])
        # Placeholder for generation logic
        # Start with simply querying the thinker model. done
        # then expand this to make an asynchronous call to the rememberer model to decide if and what from the input and output to remember
        # use a helper function to write the output to a file or database(reasearch smart approaches)
        # expand there to be a retrieval step before generation once memory storage is working.( memory retrieval) 
        # this entails asking the rememberer model for relevant memories based on the input query, then running the query, then running the memory again to see if anything needs updating
        return 
    
if __name__ == "__main__":
    think_model = "google/gemma-3-1B-it"  # Example model for thinking
    remember_model = "google/gemma-3-1B-it"

    airys = AIrysRAG(think_model, remember_model) # create an instance of our RAG llm complex
    airys.generate("What is the capital of France?")  # Example usage
    print("AIrys loaded successfully.")