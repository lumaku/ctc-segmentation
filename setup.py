from setuptools import setup, find_packages, Extension


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


try:
    from Cython.Build import cythonize
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html
ext = '.pyx' if USE_CYTHON else '.c'
extensions = [
    Extension("ctc-segmentation/ctc-segmentation-dyn",
             ["ctc-segmentation/ctc-segmentation-dyn"+ext])]
if USE_CYTHON:
    from Cython.Build import cythonize
    extensions = cythonize(extensions)

package_information = """
# CTC segmentation

CTC segmentation is used to align utterances within audio files.
It can be combined with CTC-based ASR models.
This package includes the core functions.

https://github.com/lumaku/ctc-segmentation
"""

setup(
    name="ctc_segmentation",
    version="1.0.5",

    python_requires='>=3',
    packages=find_packages(exclude=["tests"]),
    install_requires=["setuptools", "numpy", "Cython"],
    # tests_require=["torch"],
    zip_safe=False,
    ext_modules=extensions,
    cmdclass={'build_ext': build_ext},

    author="Ludwig Kuerzinger <ludwig.kuerzinger@tum.de>, "
           "Dominik Winkelbauer <dominik.winkelbauer@tum.de>",
    description="CTC segmentation to align utterances within "
                "large audio files.",
    url="https://github.com/lumaku/ctc-segmentation",

    long_description_content_type="text/markdown",
    long_description=package_information,
)
