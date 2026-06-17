.PHONY: check paper-pdf limitation-check limitation-assets limitation-reproduce limitation-reproduce-verify limitation-number-check limitation-one-result-check limitation-one-result-check-gpu limitation-one-result limitation-one-result-gpu limitation-reviewer-check-gpu

PYTHON ?= $(shell command -v python3.11 >/dev/null 2>&1 && echo python3.11 || echo python3)
VENV ?= .venv
ONE_RESULT_ARGS ?=
LIMITATION_ASSETS_ARGS ?=
LIMITATION_NUMBER_CHECK_ARGS ?=
LIMITATION_ONE_RESULT_ARGS ?=
LIMITATION_REPRODUCE_ARGS ?= --device cpu
LIMITATION_REPRODUCE_VERIFY_ARGS ?=
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
DEPS_STAMP := $(VENV)/.deps-installed

$(VENV_PYTHON):
	$(PYTHON) -m venv $(VENV)

$(DEPS_STAMP): requirements.txt | $(VENV_PYTHON)
	$(VENV_PIP) install -r requirements.txt
	touch $(DEPS_STAMP)

check: $(DEPS_STAMP)
	$(VENV_PYTHON) -m pytest -q

paper-pdf:
	$(PYTHON) scripts/render_short_paper_pdf.py

limitation-check: $(DEPS_STAMP)
	$(VENV_PYTHON) -m pytest -q tests/test_limitation_release_surface.py tests/test_limitation_short_paper_numbers.py

limitation-assets: $(DEPS_STAMP)
	HF_HUB_ENABLE_HF_TRANSFER=0 $(VENV_PYTHON) scripts/prepare_limitation_bundle.py $(LIMITATION_ASSETS_ARGS)

limitation-reproduce: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/run_limitation_paper.py --local_files_only $(LIMITATION_REPRODUCE_ARGS)

limitation-reproduce-verify: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/verify_limitation_reproduce.py $(LIMITATION_REPRODUCE_VERIFY_ARGS)

limitation-number-check: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/check_limitation_short_paper_numbers.py $(LIMITATION_NUMBER_CHECK_ARGS)

limitation-one-result-check: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/run_limitation_one_result_check.py --local_files_only $(LIMITATION_ONE_RESULT_ARGS)

limitation-one-result-check-gpu: $(DEPS_STAMP)
	$(VENV_PYTHON) scripts/run_limitation_one_result_check_gpu.py --device auto --require_accelerator --local_files_only $(LIMITATION_ONE_RESULT_ARGS)

limitation-one-result: limitation-one-result-check

limitation-one-result-gpu: limitation-one-result-check-gpu

limitation-reviewer-check-gpu:
	$(MAKE) limitation-number-check
	$(MAKE) limitation-assets
	$(MAKE) limitation-one-result-gpu
