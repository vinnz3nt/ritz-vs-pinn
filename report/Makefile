# Reusable LaTeX Makefile, a thin wrapper around latexmk.
# Usage: make pdf | make watch | make clean | make distclean | make info
# Overrides: make pdf MAIN=thesis.tex ENGINE=lualatex OUT=MyName_CV

MAIN  ?= $(firstword $(shell grep -l '^[[:space:]]*\\documentclass' *.tex 2>/dev/null))
ifeq ($(MAIN),)
  $(error No .tex file with \documentclass found; set MAIN=file.tex)
endif
JOB   := $(basename $(MAIN))
OUT   ?= $(JOB)
BUILD ?= build

# Engine priority: command line > "% !TEX program = xelatex" comment > auto-detect
ENGINE ?= $(shell sed -n 's/^%[[:space:]]*!TEX[[:space:]]*[Pp]rogram[[:space:]]*=[[:space:]]*\([a-z]*\).*/\1/p' $(MAIN) | head -n 1)
ifeq ($(ENGINE),)
  ENGINE := $(if $(shell grep -qE '^[^%]*\\usepackage.*\{(fontspec|polyglossia|unicode-math)\}' $(MAIN) && echo y),xelatex,pdflatex)
endif

# Packages that call external programs need -shell-escape
SHELL_ESCAPE ?= $(if $(shell grep -qE '^[^%]*\\usepackage.*\{(minted|svg)\}' $(MAIN) && echo y),-shell-escape)

LATEXMK := latexmk -$(patsubst pdflatex,pdf,$(ENGINE)) $(SHELL_ESCAPE) \
           -synctex=1 -file-line-error -interaction=nonstopmode -halt-on-error \
           -outdir=$(BUILD)

.PHONY: pdf suppress watch clean distclean info
.DEFAULT_GOAL := pdf

pdf:
	@find sections -type d -exec mkdir -p $(BUILD)/{} \;
	$(LATEXMK) $(MAIN)
	@cp $(BUILD)/$(JOB).pdf $(OUT).pdf
	@echo "==> $(OUT).pdf built with $(ENGINE)"

suppress:   ## build past errors (e.g. missing images); check build/$(JOB).log afterwards
	@find sections -type d -exec mkdir -p $(BUILD)/{} \;
	-$(subst -halt-on-error,-f,$(LATEXMK)) $(MAIN)
	@cp $(BUILD)/$(JOB).pdf $(OUT).pdf
	@echo "==> $(OUT).pdf built with $(ENGINE) (errors suppressed)"
	@grep -n "Error" $(BUILD)/$(JOB).log || true

watch:   ## rebuild on every save
	$(LATEXMK) -pvc -view=none -e '$$success_cmd=q/cp %D $(OUT).pdf/' $(MAIN)

clean:   ## remove aux files, keep PDFs
	latexmk -c -outdir=$(BUILD) $(MAIN)

distclean:  ## wipe build/ plus leftover clutter from older editor builds
	rm -rf $(BUILD)
	rm -f *.aux *.bbl *.blg *.bcf *.run.xml *.fdb_latexmk *.fls *.log *.out *.toc *.xdv *.synctex.gz

info:
	@echo "MAIN=$(MAIN) ENGINE=$(ENGINE) SHELL_ESCAPE=$(SHELL_ESCAPE) OUT=$(OUT).pdf"
