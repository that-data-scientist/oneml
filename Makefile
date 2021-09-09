.PHONY: setup_env
PROJECT_NAME=one-ml

ifeq (,$(shell which pyenv))
	HAS_PYENV=False
	CONDA_ROOT=$(shell conda info --root)
	BINARIES = ${CONDA_ROOT}/envs/${PROJECT_NAME}/bin
else
	HAS_PYENV=True
	CONDA_VERSION=$(shell echo $(shell pyenv version | awk '{print $$1;}') | awk -F "/" '{print $$1}')
	BINARIES = $(HOME)/.pyenv/versions/${CONDA_VERSION}/envs/${PROJECT_NAME}/bin
endif

setup_env:
ifeq (True,$(HAS_PYENV))
	@echo ">>> Detected pyenv, setting pyenv version to ${CONDA_VERSION}"
	pyenv local ${CONDA_VERSION}
	conda env create --name $(PROJECT_NAME) -f environment.yaml --force
	pyenv local ${CONDA_VERSION}/envs/${PROJECT_NAME}
else
	@echo ">>> Creating conda environment."
	conda env create --name $(PROJECT_NAME) -f environment.yaml --force
	@echo ">>> Activating new conda environment"
	source $(CONDA_ROOT)/bin/activate $(PROJECT_NAME)
endif