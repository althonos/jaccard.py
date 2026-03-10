# `jaccard.py` [![Stars](https://img.shields.io/github/stars/althonos/jaccard.py.svg?style=social&maxAge=3600&label=Star)](https://github.com/althonos/jaccard.py/stargazers)

*Utilities related to Jaccard/Tanimoto coefficients.*

[![Actions](https://img.shields.io/github/actions/workflow/status/althonos/jaccard.py/test.yml?branch=main&logo=github&style=flat-square&maxAge=300)](https://github.com/althonos/jaccard.py/actions)
[![Coverage](https://img.shields.io/codecov/c/gh/althonos/jaccard.py?style=flat-square&maxAge=3600)](https://codecov.io/gh/althonos/jaccard.py/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square&maxAge=2678400)](https://choosealicense.com/licenses/mit/)
[![PyPI](https://img.shields.io/pypi/v/jaccard.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/jaccard)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/jaccard?style=flat-square&maxAge=3600&logo=anaconda)](https://anaconda.org/bioconda/jaccard)
[![Wheel](https://img.shields.io/pypi/wheel/jaccard.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/jaccard/#files)
[![Python Versions](https://img.shields.io/pypi/pyversions/jaccard.svg?style=flat-square&maxAge=3600)](https://pypi.org/project/jaccard/#files)
[![Python Implementations](https://img.shields.io/badge/impl-universal-success.svg?style=flat-square&maxAge=3600&label=impl)](https://pypi.org/project/jaccard/#files)
[![Source](https://img.shields.io/badge/source-GitHub-303030.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/jaccard.py/)
[![GitHub issues](https://img.shields.io/github/issues/althonos/jaccard.py.svg?style=flat-square&maxAge=600)](https://github.com/althonos/jaccard.py/issues)
[![Docs](https://img.shields.io/readthedocs/jaccard/latest?style=flat-square&maxAge=600)](https://jaccard.readthedocs.io)
[![Changelog](https://img.shields.io/badge/keep%20a-changelog-8A0707.svg?maxAge=2678400&style=flat-square)](https://github.com/althonos/jaccard.py/blob/master/CHANGELOG.md)
[![Downloads](https://img.shields.io/pypi/dm/jaccard?style=flat-square&color=303f9f&maxAge=86400&label=downloads)](https://pepy.tech/project/jaccard)

## 🗺️ Overview

`jaccard.py` is a pure-Python package providing Jaccard index computation.

This library only depends on NumPy and is available for all modern Python
versions (3.6+).

### 📋 Features

Agnostic interface using duck-typing: all functions should be available for 
[NumPy] arrays, [MLX] arrays, or [PyTorch] tensors, unless noted otherwise.

[NumPy]: https://numpy.org
[MLX]: https://ml-explore.github.io/mlx/
[PyTorch]: https://pytorch.org/

The following functions are implemented:

- Jaccard similarity[\[1\]](#ref1): measure similarity between boolean vectors,
  similar to `scipy.spatial.distance.jaccard`.
- probabilistic Jaccard similarity[\[2\]](#ref2): measure similarity between
  probability vectors while quantifying uncertainty.
- centered Jaccard similarity and Jaccard testing[\[3\]](#ref3): identify
  non-random co-occurences between samples with robust statistical testing.
- collision probability Jaccard index [\[4\]](#ref4): measure similarity between 
  positive indices, using a metric that is scale invariant, sensitive to
  changes in support, and computable as a collision probability.

## 🔧 Installing

Install the `jaccard` package directly from [PyPi](https://pypi.org/project/jaccard)
which hosts universal wheels that can be installed with `pip`:
```console
$ pip install jaccard
```

<!-- Otherwise, `jaccard.py` is also available as a [Bioconda](https://bioconda.github.io/)
package:
```console
$ conda install -c bioconda jaccard
``` -->

<!-- ## 📖 Documentation

A complete [API reference](https://jaccard.readthedocs.io/en/stable/api/index.html)
can be found in the [online documentation](https://jaccard.readthedocs.io/),
or directly from the command line using
[`pydoc`](https://docs.python.org/3/library/pydoc.html):
```console
$ pydoc jaccard
``` -->

<!-- 
## 💡 Example
``` -->

## 💭 Feedback

### ⚠️ Issue Tracker

Found a bug ? Have an enhancement request ? Head over to the [GitHub issue
tracker](https://github.com/althonos/jaccard.py/issues) if you need to report
or ask something. If you are filing in on a bug, please include as much
information as you can about the issue, and try to recreate the same bug
in a simple, easily reproducible situation.

### 🏗️ Contributing

Contributions are more than welcome! See
[`CONTRIBUTING.md`](https://github.com/althonos/jaccard.py/blob/main/CONTRIBUTING.md)
for more details.

## ⚖️ License

This library is provided under the [MIT License](https://choosealicense.com/licenses/mit/).

*This project was developed by [Martin Larralde](https://github.com/althonos/) 
during his PhD project at the [Leiden University Medical Center](https://www.lumc.nl/en/) in the [Zeller team](https://zellerlab.org).*

## 📚 References

- <a id="ref1">\[1\]</a> Jaccard, P. "Étude comparative de la distribution florale dans une portion des Alpes et du Jura." Bulletin de la Société Vaudoise des Sciences Naturelles 37, 547–579 (1901). [doi:10.1111/j.1469-8137.1912.tb05611.x](https://doi.org/10.1111/j.1469-8137.1912.tb05611.x)
- <a id="ref2">\[2\]</a> Martire, I., Da Silva, P. N., Plastino, A., Fabris, F. & Freitas, A. A. "A novel probabilistic Jaccard distance measure for classification of sparse and uncertain data". Proceedings of the 5th Symposium on Knowledge Discovery, Mining and Learning, 81-88 (2017).
- <a id="ref3">\[3\]</a> Chung, N. C., Miasojedow, B., Startek, M. & Gambin, A. "Jaccard/Tanimoto similarity test and estimation methods for biological presence-absence data". BMC Bioinformatics 20, 644 (2019). [doi:10.1186/s12859-019-3118-5](https://doi.org/10.1186/s12859-019-3118-5)
- <a id="ref4">\[4\]</a> 1. Moulton, R. & Jiang, Y. "Maximally Consistent Sampling and the Jaccard Index of Probability Distributions". in 2018 IEEE International Conference on Data Mining (ICDM) 347–356 (2018). [doi:10.1109/ICDM.2018.00050](https://doi.org/10.1109/ICDM.2018.00050).
