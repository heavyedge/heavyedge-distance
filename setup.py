import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "heavyedge_distance._wasserstein",
        ["src/heavyedge_distance/_wasserstein.pyx"],
    ),
    Extension(
        "heavyedge_distance._dfd",
        ["src/heavyedge_distance/_dfd.pyx"],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level=3),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
)
