# Paper source

`main.pdf` is the built document. `main.tex` plus `sections/*.tex`, `references.bib`,
`main.bbl` and `figures/*.pdf` are its complete source.

`arxiv-v1.tar.gz` is the arXiv submission bundle: the same fourteen files, and nothing else.
Its members are byte-identical to the ones in this directory. To check that for yourself:

```bash
mkdir -p /tmp/arxiv-check && tar xzf docs/paper/arxiv-v1.tar.gz -C /tmp/arxiv-check
for m in $(tar tzf docs/paper/arxiv-v1.tar.gz); do
  cmp -s "/tmp/arxiv-check/$m" "docs/paper/$m" || echo "DIFFERS: $m"
done
```

SHA-256 of the bundle: `2339c8d6cf39878e185ec8e42d0d4c34c4b4b2abbb0a6e444fd2ba9244c40034`

## Building

The document is plain pdfLaTeX with a pre-built bibliography, so no `bibtex` pass is needed:

```bash
cd docs/paper && pdflatex main.tex && pdflatex main.tex && pdflatex main.tex
```

Three passes: after the second the document already has 76 pages with no undefined
references and no errors, but the log still asks for a rerun to settle cross-references,
and the third clears it. `main.bbl` is tracked deliberately — arXiv builds from the
`.bbl`, not from `references.bib`.
