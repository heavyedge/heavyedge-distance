.. HeavyEdge-Distance documentation master file, created by
   sphinx-quickstart on Wed Oct 22 10:46:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

********************************
HeavyEdge-Distance documentation
********************************

Plugin of :mod:`heavyedge` to compute shape distance between edge profiles.

Usage
=====

HeavyEdge-Distance is designed to be used either as a command line program or as a Python module.

To compute shape distance matrix, convert your profile data to pre-shapes and compute distance matrix.
For example, the following lines of commands convert a profile data to area-scaled pre-shape and compute Wasserstein distance matrix.

.. code-block:: bash

    heavyedge scale --type=area <profile> -o <scaled_profile>  # command from HeavyEdge package
    heavyedge dist-wasserstein --grid-num=100 <scaled_profile> -o <distance-matrix>

Command line
------------

Command lines are provided as plugins and can be invoked by:

.. code-block:: bash

    heavyedge <command>

Refer to help message of ``heavyedge`` for list of commands and their arguments.

Python module
-------------

The Python module :mod:`heavyedge_distance` provides functions to compute distance matrix in Python runtime.
Refer to :ref:`api` section for high-level interface.

Module reference
================

.. module:: heavyedge_distance

This section provides reference for :mod:`heavyedge_distance` Python module.

.. _api:

Runtime API
-----------

.. automodule:: heavyedge_distance.api
    :members:

Low-level API
-------------

.. automodule:: heavyedge_distance.wasserstein
    :members:
