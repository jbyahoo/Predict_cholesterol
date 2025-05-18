#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Cholesterol-predition
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install requirements.txt
	

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8, black, and isort (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ARISA_DSML

all:
	requirements clean lint

.PHONY: preprocess
preprocess:
	python -m ARISA_DSML.preproc

.PHONY: train
train:
	python -m ARISA_DSML.train

.PHONY: resolve
resolve:
	python -m ARISA_DSML.resolve

.PHONY: predict
predict:
	python -m ARISA_DSML.predict

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

