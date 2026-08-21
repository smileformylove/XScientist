# XScientist arXiv Manuscript

This directory contains the source for the published arXiv system report:

> XScientist: A Git-Like Research Protocol for Long-Running Autonomous Scientific Discovery

arXiv record:

- https://arxiv.org/abs/2607.12301
- arXiv: `2607.12301`
- DOI: `10.48550/arXiv.2607.12301`

Project repository:

- https://github.com/smileformylove/XScientist

The report is intended to be read together with the repository. The repository
contains the running implementation, ARA protocol schemas, CLI tools, examples,
tests, and the manuscript source in this directory.

Files:

- `main.tex`: single-file LaTeX manuscript with inline bibliography.

Published arXiv categories:

- Primary: `cs.SE`
- Cross-list: `cs.MA`

Local compile command, if a TeX distribution is installed:

```bash
tectonic main.tex
```

or:

```bash
pdflatex main.tex
pdflatex main.tex
```

Version/update notes:

- For future arXiv replacements, upload the TeX source, not a PDF generated
  from the TeX source.
- Keep `main.tex` at the root of the uploaded source package.
- Do not include build artifacts such as `.aux`, `.log`, or local PDFs.
- Verify the arXiv-generated PDF before submitting a replacement version.

Maintenance notes:

- Configure the author metadata and contact email at submission time.
- Confirm that the repository URL is visible on the title page and in the
  availability section.
- Keep README/BibTeX references aligned with arXiv: `2607.12301`.
- Re-read the limitations section and remove or soften any claim that you do
  not want to stand behind publicly.
