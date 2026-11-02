# Makefile for managing the Python virtual environment.
VENV   := .venv
PYTHON := python3
BIN    := $(VENV)/bin
STAMP  := $(VENV)/.install-stamp

.DEFAULT_GOAL := help

## help: Show this help message
.PHONY: help
help:
		@echo "Available targets:"
		@grep -E '^## ' $(MAKEFILE_LIST) | sed 's/## /  /'

## venv: Create the virtualenv (if missing) and install the project
.PHONY: venv
venv: $(STAMP)

# Create the virtualenv only if it doesn't already exist.
$(BIN)/python:
		$(PYTHON) -m venv $(VENV)
		$(BIN)/python -m pip install --upgrade pip

# Reinstall whenever pyproject.toml changes (tracked via a stamp file).
$(STAMP): $(BIN)/python pyproject.toml
		$(BIN)/pip install -e .
		@touch $(STAMP)

## shell: Print the command to activate the virtualenv
.PHONY: shell
shell:
		@echo "Run: source $(BIN)/activate"

## clean: Remove the virtualenv
.PHONY: clean
clean:
		rm -rf $(VENV)

## rebuild: Remove and recreate the virtualenv from scratch
.PHONY: rebuild
rebuild: clean venv
