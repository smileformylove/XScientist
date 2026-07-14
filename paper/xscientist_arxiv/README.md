# XScientist arXiv Manuscript

This directory contains an arXiv-ready source draft for:

> XScientist: A Git-Like Research Protocol for Long-Running Autonomous Scientific Discovery

Project repository:

- https://github.com/smileformylove/XScientist

The report is intended to be read together with the repository. The repository
contains the running implementation, ARA protocol schemas, CLI tools, examples,
tests, and the manuscript source in this directory.

Files:

- `main.tex`: single-file LaTeX manuscript with inline bibliography.

Suggested arXiv category:

- Primary: `cs.AI`
- Possible cross-list: `cs.SE`, depending on the final emphasis.

Local compile command, if a TeX distribution is installed:

```bash
tectonic main.tex
```

or:

```bash
pdflatex main.tex
pdflatex main.tex
```

arXiv upload notes:

- Upload the TeX source, not a PDF generated from the TeX source.
- Keep `main.tex` at the root of the uploaded source package.
- Do not include build artifacts such as `.aux`, `.log`, or local PDFs.
- Verify the arXiv-generated PDF before final submission.

Before submitting:

- Confirm the author spelling and contact email.
- Confirm that the repository URL is visible on the title page and in the
  availability section.
- Add an arXiv ID to the README/BibTeX only after arXiv assigns one.
- Re-read the limitations section and remove or soften any claim that you do
  not want to stand behind publicly.
