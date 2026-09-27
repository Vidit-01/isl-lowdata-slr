# Paper: low-data isolated ISL recognition

`make_tables.py` builds every table (`generated/*.tex`) and figure (`figures/*.pdf`) from the sweep outputs.
`main.tex` only `\input`s them.

```
python paper/make_tables.py --sweeps sweeps                   # graph GCNs excluded (pending rerun)
python paper/make_tables.py --sweeps sweeps --include-graph   # after the corrected graph rerun
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Open items:
- the TODOs at the top of `refs.bib`;
- the co-author list;
- the graph-model results (ST-GCN, CTR-GCN, TD-GCN, PGF-SLR). Adding them needs new text in the Models and Limitations sections, not only regenerated tables.
