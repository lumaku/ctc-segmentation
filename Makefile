

all:
	cythonize -3 ctc_segmentation/ctc_segmentation_dyn.pyx
	python -m setuptools.launch setup.py sdist

clean:
	rm ctc_segmentation/ctc_segmentation_dyn.c
	rm -rd build/ dist/ ctc_segmentation.egg-info/

upload:
	twine upload dist/*
  
