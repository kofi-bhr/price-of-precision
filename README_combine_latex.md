# LaTeX File Combiner

This script combines a LaTeX main file with all its included files (via `\input{}` commands) into a single consolidated file.

## Purpose

When working with large LaTeX documents split across multiple files using `\input{}` commands, this tool creates a single file version that:
- Makes it easier to share the document with collaborators who don't need to see the file structure
- Simplifies submission to journals or platforms that require a single file
- Helps with version tracking of the complete document

## Usage

```bash
python3 combine_latex.py /path/to/your/main.tex
```

The script will:
1. Read the main LaTeX file
2. Find all `\input{}` commands
3. Replace each with the content of the referenced file (including nested inputs)
4. Save the result as `main_combined.tex` in the same directory as the original file

## Example

For the paper "The Price of Precision":

```bash
python3 combine_latex.py /Users/kofihairralson/Programming\ Projects/microecon-paper-personal/01_full-paper/main.tex
```

This will create `/Users/kofihairralson/Programming Projects/microecon-paper-personal/01_full-paper/main_combined.tex`
