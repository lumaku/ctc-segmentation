from setuptools import setup, find_packages, Extension
import numpy
try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False


ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension("ctc_segmentation/ctc_segmentation_dyn",
              ["ctc_segmentation/ctc_segmentation_dyn"+ext],
              include_dirs=[numpy.get_include()])]
if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions, include_path=[numpy.get_include()])


setup(
    name="ctc_segmentation",
    version="1.0",
    python_requires='>=3',
    packages=find_packages(exclude=["tests"]),

    install_requires=["setuptools", "numpy", "Cython"],
    # tests_require=["torch"],
    zip_safe=False,

    ext_modules=extensions,

    author="Ludwig Kuerzinger <ludwig.kuerzinger@tum.de>, "
           "Dominik Winkelbauer <dominik.winkelbauer@tum.de>",
    description="Prvovides CTC segmentation to align utterances within "
                "large audio files.",
    url="https://github.com/lumaku/ctc-segmentation",
)
