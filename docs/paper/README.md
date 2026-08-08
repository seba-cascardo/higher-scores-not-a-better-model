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

SHA-256 of the bundle: `aaeafe412cf9462c058f48a6c508f7404e998adfc7e62454b28924aa913167bb`

## Building

The document is plain pdfLaTeX with a pre-built bibliography, so no `bibtex` pass is needed:

```bash
cd docs/paper && pdflatex main.tex && pdflatex main.tex
```

The second pass resolves cross-references. `main.bbl` is tracked deliberately — arXiv builds
from the `.bbl`, not from `references.bib`.
