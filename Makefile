DATASET_DIR=$(shell pwd)/../dataset
OUTPUT_DIR=$(DATASET_DIR)/out

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  eval               Run evaluation script"
	@echo "  tune               Run tuning script"
	@echo "  install-deps       Install dependencies"
	@echo "  build-ipfixprobe   Build ipfixprobe Docker images"
	@echo "  venv               Create a virtual environment and install dependencies"

dataset:
	./runall.sh $(DATASET_DIR) $(OUTPUT_DIR)

eval:
	./eval.sh $(OUTPUT_DIR)

tune:
	./tune.sh $(OUTPUT_DIR) gnn-tune

install-deps:
	pip install -r requirements.txt
	pip install -r requirements-torch.txt

build-ipfixprobe: data/Dockerfile.data data/Dockerfile.bin
	podman build -t ipfixprobe:latest -f data/Dockerfile.bin .
	podman build -t process-data:latest -f data/Dockerfile.data .

venv:
	rm -rf venv
	python3 -m virtualenv venv
	. venv/bin/activate
	$(MAKE) install-deps

.PHONY: help dataset eval tune install-deps build-ipfixprobe venv
