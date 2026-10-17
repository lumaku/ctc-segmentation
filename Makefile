# Makefile for CTC segmentation
# Ludwig Kürzinger, 2021

# Build sdist and wheel in an isolated environment (needs `pip install build`).
# Cython runs automatically during the build, as declared in pyproject.toml.
all:
	python -m build

clean:
	rm ctc_segmentation/ctc_segmentation_dyn.c || echo "already clean?"
	rm -rf build/ dist/ ctc_segmentation.egg-info/ || echo "already clean?"

upload:
	twine check dist/*
	twine upload dist/*

# Tests the installed package (e.g. after `make local`), like the CI does.
test:
	pytest --import-mode=importlib


# To test the various installation methods:
github:
	cd /; pip install git+https://github.com/lumaku/ctc-segmentation --user
	
pip:
	cd /; pip install ctc-segmentation --user

local:
	pip install . --user

rm:
	cd /; pip uninstall -y ctc-segmentation

