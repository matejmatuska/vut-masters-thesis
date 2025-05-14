venv:
	rm -rf venv
	python3 -m virtualenv venv
	. venv/bin/activate
	$(MAKE) install-deps

install-deps:
	pip install -r requirements.txt
	pip install -r requirements-torch.txt

build-ipfixprobe: data/Dockerfile.data data/Dockerfile.bin
	podman build -t ipfixprobe:latest -f data/Dockerfile.bin .
	podman build -t process-data:latest -f data/Dockerfile.data .
