"""
utilities functions: borrowed from pomgrenate - https://github.com/jmschrei/pomegranate
"""
from libc.math cimport log as clog
from libc.math cimport exp as cexp

DEF NEGINF = float("-inf")
DEF INF = float("inf")


cdef double logsum_pair(double x, double y) nogil:
    """
    Perform log-sum-exp on a pair of numbers in log space..  This is calculated
    as z = log( e**x + e**y ). However, this causes underflow sometimes
    when x or y are too negative. A simplification of this is thus
    z = x + log( e**(y-x) + 1 ), where x is the greater number. If either of
    the inputs are infinity, return infinity, and if either of the inputs
    are negative infinity, then simply return the other input.
    """

    if x == INF or y == INF:
        return INF
    if x == NEGINF:
        return y
    if y == NEGINF:
        return x
    if x > y:
        return x + clog(cexp(y-x) + 1)
    return y + clog(cexp(x-y) + 1)


cdef double logsumexp(double[::1] X) nogil:
    """Calculate the log-sum-exp of an array to add in log space."""

    #X = np.array(X, dtype='float64')

    cdef double x
    cdef int i, n = X.shape[0]
    cdef double y = 0.
    cdef double x_max = NEGINF

    with nogil:
        for i in range(n):
            x = X[i]
            if x > x_max:
                x_max = x

        for i in range(n):
            x = X[i]
            if x == NEGINF:
                continue

            y += cexp(x - x_max)

    return x_max + clog(y)
