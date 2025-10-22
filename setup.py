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
    ),
]

setup(
    ext_modules=cythonize(extensions),
    include_dirs=[numpy.get_include()],
    include_package_data=True,
)
