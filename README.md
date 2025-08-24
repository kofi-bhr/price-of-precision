# The Price of Precision: A Model of Optimal Bias on the Fairness-Accuracy Frontier

This repository contains the LaTeX source for two academic papers - a letter and a full paper. The articles both interrogate the trade-offs between fairness and accuracy in decision-making models.

## Getting Started

### Prerequisites

- LaTeX distribution (TeX Live or MiKTeX)
- Visual Studio Code with LaTeX Workshop extension
- Git

### Setup

1. Clone this repository
2. Open the project folder in VS Code
3. Install LaTeX Workshop extension if not already installed

## Using LaTeX Workshop

LaTeX Workshop makes editing and compiling LaTeX documents simple:

1. Open `main.tex` in VS Code
2. Use the sidebar icons to build, view, and clean the project
   - Build: "play" (▶️) icon (or Ctrl+Alt+B)
   - View PDF: 🔍 icon (or Ctrl+Alt+V)
   - Clean temporary files: 🧹 icon

## Project Structure

- `main.tex`: Main document that imports all sections
- `sections/`: Contains all paper sections as separate .tex files
- `references.bib`: Bibliography in BibTeX format

## Git Workflow

### Branches

- Name branches with clear purpose: `feature/new-model`, `fix/typos`, `revision/reviewer1`
- Always create branches from the latest `main`

### Commits

- Write clear commit messages: "Add robustness checks to section 5"
- Commit often with logical changes

### Pull Requests

1. Push your branch to GitHub
2. Create a PR with a descriptive title
3. Summarize changes in the description
4. Request review from collaborators

## Build Files

The `.gitignore` file prevents LaTeX build files from being committed. Ignored files include:

- `*.aux`, `*.log`, `*.synctex.gz`, `*.bbl`, `*.blg`
- `*.fdb_latexmk`, `*.fls`, `*.out`, `*.ttt`, `*.fff`

Do not commit these files.

## Contribution Guidelines

### Editing Sections

1. Work in the appropriate section file in `sections/`
2. Use comments for major changes: `% KHR: Changed model assumption here`
3. Use initials to mark your contributions

### Math Notation

- Define new notation in the preamble of `main.tex`
- Use consistent notation throughout

### Figures and Tables

- Store figures in a logical folder structure
- Use descriptive filenames
- Include source files for graphs when possible

## Questions?

Contact Kofi Hair-Ralston at kofibhairralston@gmail.com