import os
import argparse

def combine_txt_files(input_dir, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for filename in os.listdir(input_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(input_dir, filename)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        outfile.write(infile.read())
                        outfile.write('\n')  # Add a newline between files
        print(f"All .txt files in '{input_dir}' have been combined into '{output_file}'.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Combine all .txt files in a directory into one file.")
    parser.add_argument("input_dir", help="The directory containing .txt files to combine.")
    parser.add_argument("output_file", help="The output file to save the combined content.")
    args = parser.parse_args()

    combine_txt_files(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()