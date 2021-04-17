# Makefile for CTC segmentation
# Ludwig Kürzinger, 2021

all:
	cythonize -3 ctc_segmentation/ctc_segmentation_dyn.pyx
	python setup.py sdist

clean:
	rm ctc_segmentation/ctc_segmentation_dyn.c || echo "already clean?"
	rm -rf build/ dist/ ctc_segmentation.egg-info/ || echo "already clean?"

upload:
	twine upload dist/*
  
test:
	cd tests; python -c "import test_ctc_segmentation as test; test.test_ctc_segmentation()"
	cd tests; python -c "import test_ctc_segmentation as test; test.test_determine_utterance_segments()"
	cd tests; python -c "import test_ctc_segmentation as test; test.test_prepare_text()"
	cd tests; python -c "import test_ctc_segmentation as test; test.test_prepare_tokenized_text()"
	cd tests; python -c "import test_ctc_segmentation as test; test.test_prepare_token_list()"

github:
	cd /; pip install git+https://github.com/lumaku/ctc-segmentation --user
	
pip:
	cd /; pip install ctc-segmentation --user

rm:
	cd /; pip uninstall -y ctc-segmentation

