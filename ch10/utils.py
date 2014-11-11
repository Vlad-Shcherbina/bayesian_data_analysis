from __future__ import division


def strict_zip(*xss):
    for ys in xss[1:]:
        assert len(xss[0]) == len(ys)
    return zip(*xss)


def factorial(n):
    assert n >= 0
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def binomial(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))
