__author__ = "Martin Larralde <martin.larralde@embl.de>"
__version__ = "0.1.0-dev.1"
__license__ = "MIT"

# Small addition to the docstring: we want to show a link redirecting to the
# rendered version of the documentation, but this can only work when Python
# is running with docstrings enabled
if __doc__ is not None:
    __doc__ += """See Also:
    An online rendered version of the documentation for this version
    of the library on
    `Read The Docs <https://jaccard.readthedocs.io/en/v{}/>`_.

    """.format(__version__)


def similarity(u, v, w=None):
    r"""
    Compute the Jaccard similarity index between two boolean vectors.

    :math:`u` and :math:`v` are expected to be multi-dimensional objects
    of the same shape, with the last dimension treated as boolean vectors.
    In the one-dimensional case, where :math:`u` and :math:`v` are boolean
    vectors :math:`u \equiv (u_1, \cdots, u_n)` and
    :math:`v \equiv (v_1, \cdots, v_n)` that are not both zero, their
    *Jaccard dissimilarity* is defined as ([1]_, p. 26)

    .. math::

       d_\textrm{jaccard}(u, v) := \frac{c_{11}}
                                        {c_{11} + c_{10} + c_{01}}

    where

    .. math::

       c_{ij} := \sum_{1 \le k \le n, u_k=i, v_k=j} 1

    for :math:`i, j \in \{ 0, 1\}`.  If :math:`u` and :math:`v` are both zero,
    their Jaccard dissimilarity is defined to be zero. [2]_

    If a (non-negative) weight vector :math:`w \equiv (w_1, \cdots, w_n)`
    is supplied, the *weighted Jaccard dissimilarity* is defined similarly
    but with :math:`c_{ij}` replaced by

    .. math::

       \tilde{c}_{ij} := \sum_{1 \le k \le n, u_k=i, v_k=j} w_k

    Parameters
    ----------
    u : (*, N) array_like of bools
        Input vector.
    v : (*, N) array_like of bools
        Input vector.
    w : (*, N) array_like of floats, optional
        Weights for each pair of :math:`(u_k, v_k)`.  Default is ``None``,
        which gives each pair a weight of ``1.0``.

    Returns
    -------
    jaccard : (*,)
        The Jaccard similarity between vectors `u` and `v`, optionally
        weighted by `w` if supplied.

    Notes
    -----
    The Jaccard dissimilarity satisfies the triangle inequality and is
    qualified as a metric. [2]_

    The *Jaccard index*, or *Jaccard similarity coefficient*, is equal to
    one minus the Jaccard dissimilarity. [3]_

    The dissimilarity between general (finite) sets may be computed by
    encoding them as boolean vectors and computing the dissimilarity
    between the encoded vectors.
    For example, subsets :math:`A,B` of :math:`\{ 1, 2, ..., n \}` may be
    encoded into boolean vectors :math:`u, v` by setting
    :math:`u_k := 1_{k \in A}`, :math:`v_k := 1_{k \in B}`
    for :math:`k = 1,2,\cdots,n`.

    References
    ----------
    .. [1] Kaufman, L. and Rousseeuw, P. J.  (1990).  "Finding Groups in Data:
           An Introduction to Cluster Analysis."  John Wiley & Sons, Inc.
    .. [2] Kosub, S.  (2019).  "A note on the triangle inequality for the
           Jaccard distance."  *Pattern Recognition Letters*, 120:36-38.
    .. [3] https://en.wikipedia.org/wiki/Jaccard_index

    Examples
    --------
    >>> import jaccard
    >>> from numpy import array

    Non-zero vectors with no matching 1s have similarity of 0.0:

    >>> jaccard.similarity(array([1, 0, 0]), array([0, 1, 0]))
    0.0

    Vectors with some matching 1s have similarity greater than 0.0:

    >>> jaccard.similarity(array([1, 0, 0, 0]), array([1, 1, 1, 0]))
    0.3333333333333333

    Identical vectors, including zero vectors, have similarity of 1.0:

    >>> jaccard.similarity(array([1, 0, 0]), array([1, 0, 0]))
    1.0
    >>> jaccard.similarity(array([0, 0, 0]), array([0, 0, 0]))
    1.0

    Arrays of higher dimensions are supported and will be reduced along
    their last dimension:

    >>> jaccard.similarity(
    ...    array([[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
    ...    array([[1, 0, 0], [1, 1, 0], [0, 1, 0]]),
    ... )
    array([1. , 0.5, 0. ])

    The following example computes the similarity from a confusion matrix
    directly by setting the weight vector to the frequency of True Positive,
    False Negative, False Positive, and True Negative:

    >>> jaccard.similarity(
    ...    array([1, 1, 0, 0]),
    ...    array([1, 0, 1, 0]),
    ...    w=array([31, 41, 59, 26])
    ... )
    0.2366412213740458

    """
    if u.ndim != v.ndim:
        raise ValueError("shape mismatch")
    elif u.shape[-1] != v.shape[-1]:
        raise ValueError("dimension mismatch")

    nzu = u != 0
    nzv = v != 0

    inter = nzu & nzv
    union = nzu | nzv

    if w is not None:
        inter = w * inter
        union = w * union

    a = inter.sum(axis=-1)
    b = union.sum(axis=-1)

    if len(u.shape) > 1:
        out = a / b.clip(min=1)
        out[b == 0] = 1.0
    else:
        out = 1.0 if b == 0 else a / b

    return out


def probabilistic_similarity(u, v):
    r"""
    Compute the probabilistic Jaccard similarity between two vectors.

    In the one-dimensional case, where :math:`u` and :math:`v` are
    probability vectors :math:`u \equiv (u_1, \cdots, u_n)` and
    :math:`v \equiv (v_1, \cdots, v_n)` that are not both zero, their
    *probabilistic Jaccard dissimilarity* is defined as ([1]_, p. 83)

    .. math::

        d_\textrm{p. jaccard}(u, v) := \frac{\sum_{i=1}^{n}{u_i v_i}}
                                            {\sum_{i=1}^{n}{u_i + v_i - u_i v_i}}

    which can be expressed with the dot-product :math:`\cdot` and the
    :math:`L_1` norm as

    .. math::

        d_\textrm{p. jaccard}(u, v) := \frac{u \cdot v}
                                            {\|u\| + \|v\| + u \cdot v}

    Parameters
    ----------
    u : (*, N) array_like of floats
        Input vector.
    v : (*, N) array_like of floats
        Input vector.

    Returns
    -------
    probjaccard: (*,)
        The probabilistic Jaccard similarity between vectors `u` and `v`.

    References
    ----------
    .. [1] Martire, I., da Silva, P. N., Plastino, A., Fabris, F., and
           Freitas, A. A. (2017) "A novel probabilistic Jaccard distance
           measure for classification of sparse and uncertain data."
           Proceedings of the 5th Symposium on Knowledge Discovery, Mining
           and Learning (KDMiLe): 81-88.


    Examples
    --------
    >>> import jaccard
    >>> from numpy import array

    Probabilistic Jaccard similarity between boolean vectors equals to their
    Jaccard similarity:

    >>> jaccard.similarity(array([1, 0, 0]), array([1, 1, 0]))
    0.5
    >>> jaccard.probabilistic_similarity(array([1, 0, 0]), array([1, 1, 0]))
    0.5

    Vectors with some matching positions have similarity greater than 0.0:

    >>> jaccard.probabilistic_similarity(array([0.4, 0, 0.8]), array([1, 0, 1]))
    0.6...

    Arrays of higher dimensions are supported and will be reduced along
    their last dimension:

    >>> jaccard.probabilistic_similarity(
    ...    array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0.5, 1, 0]]),
    ...    array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 0.5]]),
    ... )
    array([1. , 0.5, 0. , 0.5])

    """
    if u.ndim != v.ndim:
        raise ValueError("shape mismatch")
    elif u.shape[-1] != v.shape[-1]:
        raise ValueError("dimension mismatch")

    if u.ndim == 1:
        tt = u @ v
    else:
        tt = (u * v).sum(axis=-1)

    return tt / (u.sum(axis=-1) - tt + v.sum(axis=-1))


def probabilistic_similarity_pairwise(X, Y=None):
    r"""
    Compute the probabilistic Jaccard similarity between pairs of vectors.

    A fast approach for computing pairwise probabilistic similarity
    between two matrices of observations :math:`A \in \mathbb{B}^{m \times n}`
    can be computed with:

    .. math::

        s_{i,j} = b_{i,j} / c_{i,j}

    where:

    .. math::

        b_{i, j} = \|X_i \cap X_j\| = \sum_{k}{a_{k,i} a_{k,j}}
    
    so :math:`B = A^T A`, and:

    .. math::

        c_{i, j} = \|X_i \cup X_j\| = \|X_i\| + \|X_j\| - \|X_i \cap X_j\| 
                 = \sum_{k}{a_{k,i}} + \sum_{k}{a_{k,j}} - b_{i,j}

    Parameters
    ----------
    X : (M, d) array_like of floats
        Input matrix.
    Y : (N, d) array_like of floats
        Input matrix.

    Returns
    -------
    probjaccard: (M,N)
        The probabilistic Jaccard similarity matrix between row vectors of
        `X` and `Y`, pairwise.

    References
    ----------
    .. [1] Martire, I., da Silva, P. N., Plastino, A., Fabris, F., and
           Freitas, A. A. (2017) "A novel probabilistic Jaccard distance
           measure for classification of sparse and uncertain data."
           Proceedings of the 5th Symposium on Knowledge Discovery, Mining
           and Learning (KDMiLe) 81-88.
    .. [2] Besta, M., Kanakagiri, R., Mustafa, H., Karasikov, M., Rätsch, G.
           & Hoefler, T. (2020). "Communication-Efficient Jaccard similarity 
           for High-Performance Distributed Genome Comparisons". *in* 2020 
           IEEE International Parallel and Distributed Processing Symposium 
           (IPDPS) 112-1132. :doi:`10.1109/IPDPS47924.2020.00118`.

    Examples
    --------
    >>> import jaccard
    >>> from numpy import array

    >>> jaccard.probabilistic_similarity_pairwise(
    ...    array([[1, 0, 0], [0, 1, 0]]),
    ...    array([[0.8, 0.4, 0], [0.6, 1, 0], [0, 0.3, 0], [0, 1, 0.5]]),
    ... )
    array([[0.5714..., 0.3       , 0.        , 0.        ],
           [0.2222..., 0.625     , 0.3       , 0.66666...]])

    """
    if Y is None:
        Y = X

    if X.ndim != Y.ndim:
        raise ValueError("shape mismatch")
    elif X.shape[-1] != Y.shape[-1]:
        raise ValueError("dimension mismatch")
    elif X.ndim < 2:
        raise ValueError("data needs at least two dimensions")

    tt = X @ Y.transpose(-1, -2)
    return tt / (X.sum(axis=-1).reshape(-1, 1) - tt + Y.sum(axis=-1).reshape(1, -1))


def centered_similarity(u, v, w=None, *, pu=None, pv=None):
    r"""
    Compute the centered Jaccard similarity between two boolean vectors.

    Parameters
    ----------
    u : (*, N) array_like of floats
        Input vectors.
    v : (*, N) array_like of floats
        Input vectors.
    w : (*, N) array_like of floats, optional
        Weights for each pair of :math:`(u_k, v_k)`.  Default is ``None``,
        which gives each pair a weight of ``1.0``.

    References
    ----------
    .. [1] Chung, N. C., Miasojedow, B., Startek, M. & Gambin, A. (2019)
           "Jaccard/Tanimoto similarity test and estimation methods for
           biological presence-absence data". BMC Bioinformatics 20, 644.

    Examples
    --------
    >>> import jaccard
    >>> from numpy import array

    Zero vectors are supported but the probabilities for each vector need to
    be given *a priori*:

    >>> jaccard.centered_similarity(array([0, 0]), array([0, 0]), pu=0.5, pv=0.5)
    0.0

    Identical vectors may have a centered similarity lower than one since
    they account for the probability of the events
    (here :math:`P(u) = P(v) = \frac13`):

    >>> jaccard.centered_similarity(array([1, 0, 0]), array([1, 0, 0]))
    0.8

    Negatively correlated samples have a negative similarity:

    >>> jaccard.centered_similarity(array([1, 0, 0, 1]), array([0, 1, 1, 0]))
    -0.3333...

    """
    if u.ndim != v.ndim:
        raise ValueError("shape mismatch")
    elif u.shape[-1] != v.shape[-1]:
        raise ValueError("dimension mismatch")

    nzu = u != 0
    nzv = v != 0

    if pu is None:
        pu = nzu.sum(axis=-1) / nzu.shape[-1]
    if pv is None:
        pv = nzv.sum(axis=-1) / nzv.shape[-1]
    if pu == 0 or pv == 0:
        raise ValueError("zero probabilities")

    inter = nzu & nzv
    union = nzu | nzv

    if w is not None:
        inter = w * inter
        union = w * union

    a = inter.sum(axis=-1)
    b = union.sum(axis=-1)

    E = (pu * pv) / (pu + pv - pu*pv)

    if len(u.shape) > 1:
        T = a / b.clip(min=1)
        T[b == 0] = E
    else:
        T = E if b == 0 else a / b

    return T - E


def collision_probability(u, v, *, dtype=None):
    r"""
    Compute the Jaccard collision probability from two positive vectors.

    The Jaccard index is defined for boolean vectors which indicate set
    membership. Several generalizations have been proposed to extend it
    to arbitrary vectors. One of such generalization was
    proposed by Moulton & Jiang, which treats its input as a
    probability distribution. [1]_

    The following formulation expresses the Jaccard index between
    two positive vectors as a collision probability through MinHashing:

    .. math::

       J_{\mathcal{P}}(u, v) := \sum_{i | u_i, v_i > 0 }
            {\frac{1}{\sum_{j}{ max(\frac{u_j}{u_i}, \frac{v_j}{v_i}) }}}

    For discrete uniform distributions, the collision probability is equal
    to the weighted Jaccard index.

    References
    ----------
    .. [1] Moulton, R., & Jiang, Y. (2018)
           "Maximally Consistent Sampling and the Jaccard Index of Probability
           Distributions". ICDM Workshop on High Dimensional Data Mining.

    Examples
    --------
    >>> import jaccard
    >>> from numpy import array

    Collision probability between discrete vectors is equal to their
    Jaccard similarity:

    >>> jaccard.collision_probability(array([1, 0, 0]), array([1, 1, 0]))
    0.5

    Collision probability is scale invariant, i.e. 
    :math:`J_{\mathcal{P}}(\alpha u, v) = J_{\mathcal{P}}(u, v)` :

    >>> jaccard.collision_probability(array([1, 0, 0]), array([2, 2, 0]))
    0.5

    """
    if u.ndim != v.ndim:
        raise ValueError("shape mismatch")
    elif u.shape[-1] != v.shape[-1]:
        raise ValueError("dimension mismatch")
    elif u.ndim > 1:
        raise ValueError("data can only be one dimensional")

    # iterate (fixme! O(nlog(n))!)
    out = 0.0
    for i in range(u.shape[-1]):
        if u[i] == 0 or v[i] == 0:
            continue
        out += 1 / (u / u[i]).clip(min=v / v[i]).sum()
    return out
