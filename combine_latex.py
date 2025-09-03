#!/usr/bin/env python3
"""
Script to combine a LaTeX main file with all its input files into a single file.
This removes all \input{} commands and replaces them with the actual content.
"""

import os
import re
import sys

def read_file(file_path):
    """Read a file and return its content as a string."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def process_input_command(match, base_dir):
    """Process an \input{} command and return the content of the referenced file."""
    # Extract the file path from the \input{} command
    file_path = match.group(1)
    
    # If the file doesn't have a .tex extension, add it
    if not file_path.endswith('.tex'):
        file_path += '.tex'
    
    # Create the full path
    full_path = os.path.join(base_dir, file_path)
    
    # Read the content of the referenced file
    content = read_file(full_path)
    
    if content is None:
        # If we couldn't read the file, return the original \input{} command
        return match.group(0)
    
    # Process any nested \input{} commands in the content
    return process_inputs(content, os.path.dirname(full_path))

def process_inputs(content, base_dir):
    """Replace all \input{} commands in the content with the content of the referenced files."""
    # Regular expression to match \input{...} commands
    input_pattern = r'\\input\{([^}]+)\}'
    
    # Replace all \input{} commands
    return re.sub(input_pattern, lambda match: process_input_command(match, base_dir), content)

def combine_latex_files(main_file_path, output_file_path):
    """Combine a LaTeX main file with all its input files into a single file."""
    # Read the main file
    main_content = read_file(main_file_path)
    if main_content is None:
        return False
    
    # Get the directory of the main file
    main_dir = os.path.dirname(main_file_path)
    
    # Process all \input{} commands
    combined_content = process_inputs(main_content, main_dir)
    
    # Write the combined content to the output file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(combined_content)
        return True
    except Exception as e:
        print(f"Error writing to file {output_file_path}: {e}")
        return False

def main():
    """Main function."""
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python combine_latex.py <path_to_main_tex_file>")
        print("The combined file will be saved as <main_file_name>_combined.tex")
        return
    
    # Get the main file path from the command line arguments
    main_file_path = sys.argv[1]
    
    # Create the output file path
    output_file_path = os.path.splitext(main_file_path)[0] + "_combined.tex"
    
    # Combine the LaTeX files
    success = combine_latex_files(main_file_path, output_file_path)
    
    if success:
        print(f"Successfully combined LaTeX files into {output_file_path}")
    else:
        print("Failed to combine LaTeX files")

if __name__ == "__main__":
    main()
