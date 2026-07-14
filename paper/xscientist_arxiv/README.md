# XScientist arXiv Manuscript

This directory contains an arXiv-ready source draft for:

> XScientist: A Long-Running Autonomous Scientific Research System

Files:

- `main.tex`: single-file LaTeX manuscript with inline bibliography.

Suggested arXiv category:

- Primary: `cs.AI`
- Possible cross-list: `cs.SE`, depending on the final emphasis.

Local compile command, if a TeX distribution is installed:

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

- Confirm the author spelling, affiliation, and optional contact email.
- Add an arXiv ID to the README/BibTeX only after arXiv assigns one.
- Re-read the limitations section and remove or soften any claim that you do
  not want to stand behind publicly.
