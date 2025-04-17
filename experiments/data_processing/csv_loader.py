import csv
import os

def csv_to_text(input_csv, txt_file):
    """
    Extracts the 'content' and 'author.global_name' columns from a CSV file
    and writes them to a non-delimited .txt file.

    Args:
        input_csv (str): Path to the input CSV file.
        txt_file (file object): Opened file object for writing the output.

    Returns:
        int: The number of words written to the text file.
    """
    word_count = 0
    try:
        with open(input_csv, 'r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                # Extract the desired columns and write them to the text file
                content = row.get('Content', '')
                txt_file.write(f"{content}\n")
                word_count += len(content.split())
    except Exception as e:
        print(f"An error occurred while processing {input_csv}: {e}")
    return word_count

def process_all_csv_files(directory, prefix):
    """
    Processes all CSV files in a directory with a given prefix and combines their content
    into a single text file.

    Args:
        directory (str): Path to the directory containing the CSV files.
        prefix (str): Prefix of the CSV files to process.
    """
    output_file = os.path.join(directory, f"{prefix}_combined.txt")
    total_word_count = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as txt_file:
            for filename in os.listdir(directory):
                if filename.startswith(prefix) and filename.endswith('.csv'):
                    input_csv = os.path.join(directory, filename)
                    word_count = csv_to_text(input_csv, txt_file)
                    total_word_count += word_count
                    print(f"Processed {input_csv} with {word_count} words.")
        print(f"Combined output written to {output_file} with {total_word_count} total words.")
    except Exception as e:
        print(f"An error occurred while processing files in {directory}: {e}")

if __name__ == "__main__":
    # Example usage
    input_directory = input("Enter the path to the directory containing the CSV files: ").strip()
    file_prefix = input("Enter the prefix of the CSV files to process: ").strip()
    process_all_csv_files(input_directory, file_prefix)