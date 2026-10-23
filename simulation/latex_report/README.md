# LaTeX midterm report

`main.tex` is the editable report, with its IEEE-style references included at the
end. `main.pdf` is compiled directly from that source. The author field is populated from the project README. The student ID field remains
blank and must be completed before submitting the individual report.

The report uses a single column, 11-point type, 1.5 line spacing, and one-inch
margins. The course's midterm limit is 5–10 report-body pages, excluding the title page and references. The compiled report has 10 body pages, one title page, and one references page.

## Compile

Open this folder as a LaTeX project, or upload the extracted package to Overleaf
and choose `main.tex` as the main document. From this folder, use either:

```sh
tectonic main.tex
```

or a standard LaTeX installation:

```sh
pdflatex main.tex
pdflatex main.tex
```

The figures use relative paths; keep the folder structure intact. No BibTeX
step is needed.

## Supporting material

- `figures/`: all eight images used in the report.
- `simulalted_environments/`: the Normal/Fog/Rain video, the dense/bright versus
  sparse/weak comparison, and the four-case sampling/reflectance comparison,
  together with their source manifests.
- `visual_report.html`: an offline interactive companion with the videos and
  saved measurement evidence. Open it directly in a browser.
- `simulation/`: Python source, configurations, tests, the existing project
  guide, and saved numerical evidence. Follow its `README.md` to install and
  regenerate the simulation, figures, and media. Large raw navigation histories
  regenerate from the fixed seeds and are omitted from the ZIP.
- `manifest.json`: the package inventory, file checksums, and requirement-to-
  evidence guide.

Videos are companion files rather than embedded PDF video streams. Open them
directly or through the interactive companion.

## Rebuild the package from the repository

From the `simulation/` project directory:

```sh
python tools/package_latex_report.py --stage-only
cd latex_report
tectonic main.tex
cd ..
python tools/package_latex_report.py
```

The archive is written to `outputs/submission/Midterm_Project_LaTeX.zip`.
