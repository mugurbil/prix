PYTHON ?= python3

.PHONY: test paper-figures accuracy-sweep paper

test:
	JAX_PLATFORMS=cpu $(PYTHON) -m pytest tests/ -q

paper-figures:
	$(PYTHON) benchmarks/plot_results.py
	$(PYTHON) benchmarks/transport_ablation.py benchmarks/results/cpu_amd_epyc_7b12.json --fig docs/figures/transport_ablation

accuracy-sweep:
	JAX_PLATFORMS=cpu $(PYTHON) benchmarks/accuracy_sweep.py

paper:
	cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex && cp main.pdf prix_arxiv.pdf && rm -f main.aux main.blg main.out main.pdf main.log
