# coding: utf-8
"""Test doctest contained tests in every file of the module."""

import configparser
import doctest
import importlib
import json
import gzip
import os
import pkgutil
import re
import shutil
import sys
import types
import warnings
import numpy
from unittest import mock

import jaccard


def _load_tests_from_module(tests, module, globs, setUp=None, tearDown=None):
    """Load tests from module, iterating through submodules."""
    for attr in (getattr(module, x) for x in dir(module) if not x.startswith("_")):
        if isinstance(attr, types.ModuleType):
            suite = doctest.DocTestSuite(
                attr,
                globs,
                setUp=setUp,
                tearDown=tearDown,
                optionflags=+doctest.ELLIPSIS,
            )
            tests.addTests(suite)
    return tests


def load_tests(loader, tests, ignore):
    """`load_test` function used by unittest to find the doctests."""
    _current_cwd = os.getcwd()
    _options =  numpy.get_printoptions()

    def setUp(self):
        warnings.simplefilter("ignore")
        if numpy.__version__[0] != '1':
            numpy.set_printoptions(legacy="1.25")
        # os.chdir(os.path.realpath(os.path.join(__file__, os.path.pardir, "data")))

    def tearDown(self):
        # os.chdir(_current_cwd)
        numpy.set_printoptions(**_options)
        warnings.simplefilter(warnings.defaultaction)

    # doctests are not compatible with `green`, so we may want to bail out
    # early if `green` is running the tests
    if sys.argv[0].endswith("green"):
        return tests

    # recursively traverse all library submodules and load tests from them
    packages = [None, "jaccard"]
    for pkg in iter(packages.pop, None):

        module = importlib.import_module(pkg)
        globs = dict(jaccard=jaccard, **module.__dict__)
        tests.addTests(
            doctest.DocTestSuite(
                module,
                globs=globs,
                setUp=setUp,
                tearDown=tearDown,
                optionflags=+doctest.ELLIPSIS,
            )
        )

        for _, subpkgname, subispkg in pkgutil.walk_packages(module.__path__):
            # if the submodule is a package, we need to process its submodules
            # as well, so we add it to the package queue
            if subispkg and subpkgname != "tests":
                packages.append(subpkgname)

    return tests
