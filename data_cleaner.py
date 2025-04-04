import re
import sys
import os

def clean_text_file(input_file, output_file):
    """
    Cleans a text file by removing all non-alphanumeric characters, hyperlinks, 
    and excess spaces.
    
    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to save the cleaned text file.
    """
    try:
        with open(input_file, 'r') as infile:
            content = infile.read()
        
        # Remove hyperlinks
        content = re.sub(r'http[s]?://\S+|www\.\S+', '', content)
        
        # Remove non-alphanumeric characters (excluding spaces)
        content = re.sub(r'[^a-zA-Z0-9\s]', '', content)
        
        # Remove excess spaces
        cleaned_content = re.sub(r'\s+', ' ', content).strip()
        
        with open(output_file, 'w') as outfile:
            outfile.write(cleaned_content)
        
        print(f"File cleaned and saved as {output_file}.")
    except FileNotFoundError:
        print("Error: The input file does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data_cleaner.py <input_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.isfile(input_file):
        print("Error: The specified file does not exist.")
        sys.exit(1)
    
    # Generate output file name
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_cleaned{ext}"
    
    clean_text_file(input_file, output_file)