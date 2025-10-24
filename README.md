# HeavyEdge-Distance

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/heavyedge-distance.svg)](https://pypi.python.org/pypi/heavyedge-distance/)
[![PyPI Version](https://img.shields.io/pypi/v/heavyedge-distance.svg)](https://pypi.python.org/pypi/heavyedge-distance/)
[![License](https://img.shields.io/github/license/heavyedge/heavyedge-distance)](https://github.com/heavyedge/heavyedge-distance/blob/master/LICENSE)
[![CI](https://github.com/heavyedge/heavyedge-distance/actions/workflows/ci.yml/badge.svg)](https://github.com/heavyedge/heavyedge-distance/actions/workflows/ci.yml)
[![CD](https://github.com/heavyedge/heavyedge-distance/actions/workflows/cd.yml/badge.svg)](https://github.com/heavyedge/heavyedge-distance/actions/workflows/cd.yml)
[![Docs](https://readthedocs.org/projects/heavyedge-distance/badge/?version=latest)](https://heavyedge-distance.readthedocs.io/en/latest/?badge=latest)

Package to compute shape distance between edge profiles.

## Usage

Heavyedge-Distance provides dataset classes profile data file.

Refer to the package documentation for more information.

## Installation

```
$ pip install heavyedge-distance
```

## Documentation

The manual can be found online:

> https://heavyedge-distance.readthedocs.io

If you want to build the document yourself, get the source code and install with `[doc]` dependency.
Then, go to `doc` directory and build the document:

```
$ pip install .[doc]
$ cd doc
$ make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.

## Developing

### Installation

For development features, you must install the package by `pip install -e .[dev]`.

### Testing

Run `pytest` command to perform unit test.

When doctest is run, buildable sample data are rebuilt by default.
To disable this, set `HEAVYEDGE_TEST_REBUILD` environment variable to zero.
For example,
```
HEAVYEDGE_TEST_REBUILD=0 pytest
```
