from __future__ import division

import numpy
import numpy.random
numpy.random.seed(42)

import matplotlib.pyplot as plt


from utils import *


def logit(p):
    return numpy.log(p) - numpy.log(1 - p)


def inv_logit(x):
    ex = numpy.exp(x)
    return ex / (1 + ex)


def make_array(make_element, size):
    result = numpy.zeros(size)
    for x in numpy.nditer(result, op_flags=['readwrite']):
        x[...] = make_element()
    return result


def poisson_plus(lam, size=None):
    if size is not None:
        return make_array(lambda: poisson_plus(lam), size)

    while True:
        result = numpy.random.poisson(lam)
        if result > 0:
            return result


class Params(object):
    def __str__(self):
        return 'Params({})'.format(','.join(
            '\n    {}={}'.format(k, v)
            for k, v in sorted(self.__dict__.items())))

DATASET_SIZE = 1
J = 10
POISSON_LAMBDA = 5

def generate_params():
    params = Params()

    params.x = numpy.random.uniform(0, 1, size=J)
    params.n = poisson_plus(POISSON_LAMBDA, size=J).astype(int)

    params.alpha = numpy.random.standard_t(df=4) * 2
    params.beta = numpy.random.standard_t(df=4)

    params.theta = inv_logit(params.alpha + params.beta * params.x)

    return params


def generate_sample(params):
    return numpy.vectorize(numpy.random.binomial)(params.n, params.theta)


def likelihood(params, y):
    p = 1
    for y_j, theta_j, n_j in strict_zip(y, params.theta, params.n):
        if y_j > n_j:
            return 0
        p *= binomial(n_j, y_j) * theta_j ** y_j * (1 - theta_j) ** (n_j - y_j)
    return p


def likelihood_marginalized_over_n(params, y):
    prob_zero = numpy.exp(-POISSON_LAMBDA)
    p = 1
    for y_j, theta_j in strict_zip(y, params.theta):
        lam = POISSON_LAMBDA * theta_j
        pp = lam ** y_j / factorial(y_j) * numpy.exp(-lam)
        if y_j == 0:
            assert pp >= prob_zero
            pp -= prob_zero
        p *= pp
    return p


def estimate_acceptance_probability(true_params, samples, likelihood_fn):
    N = 200
    result = 0
    for _ in range(N):
        params = generate_params()
        p = 1.0
        for sample in samples:
            p *= likelihood_fn(params, sample)
        assert p <= 1.0
        result += p / N
    result = max(result, 1e-50)  # to avoid divisions by zero and stuff
    return result


def plot_acceptance_probabilities():
    aps = []
    for likelihood_fn in likelihood, likelihood_marginalized_over_n:
        acceptance_probabilities = []

        for _ in range(300):
            print _
            true_params = generate_params()
            samples = [generate_sample(true_params) for _ in range(DATASET_SIZE)]
            acceptance_probabilities.append(
                estimate_acceptance_probability(
                    true_params, samples, likelihood_fn))

        aps.append(acceptance_probabilities)

    min_ap = min(min(ap) for ap in aps)
    bins = numpy.logspace(
        numpy.log(min_ap) / numpy.log(10), 0, 30)
    plt.hist(
        aps, bins=bins,
        label=['using likelihood', 'using likelihood_marginalized_over_n'])
    plt.legend()
    plt.xscale('log')
    plt.show()


def plot_posterior(likelihood_fn=likelihood_marginalized_over_n):
    true_params = generate_params()
    samples = [generate_sample(true_params) for _ in range(DATASET_SIZE)]

    print 'true params:', true_params
    print 'samples:', samples

    # plot prior
    for _ in range(500):
        params = generate_params()
        plt.plot([params.alpha], [params.beta], 'g.')

    # plot posterior
    accepted = 0
    N = 10 ** 5
    for _ in range(N):
        if _ % 10000 == 0:
            print int(100 * _ / N), '%'
            print accepted, 'samples so far'
        params = generate_params()
        p = 1.0
        for sample in samples:
            p *= likelihood_fn(params, sample)
        assert p <= 1.0
        if numpy.random.uniform(0, 1) < p:
            plt.plot([params.alpha], [params.beta], 'r.')
            accepted += 1
            if accepted > 500:
                break

    # plot true values
    plt.plot([true_params.alpha], [true_params.beta], 'k*', markersize=20)

    plt.show()


def main():
    plot_posterior()
    #plot_acceptance_probabilities()
    return


if __name__ == '__main__':
    main()
