# Acceptance rate distribution

Method: draw alpha, beta, and nuisance parameters from prior, probabilistically accept them according to the likelihood. 

The problem is that likelihood could be small, so most of the samples will be rejected.

The plot below displays distribution of acceptance probability for randomly drawn true alpha and beta.

Two methods are used:
* raw likelihood
* likelihood marginalized over n

It seems second method in most cases is slightly worse than the first one. 
Its crucial advantage is that it does not appear to have pathological parameter values where acceptance rate is extremely low.

There is a cutoff at ~10^-50 for convenience, so leftmost bucket includes everything below this threshold as well.

![distribution of acceptance probability](https://raw.githubusercontent.com/Vlad-Shcherbina/bayesian_data_analysis/master/ch10/distribution_of_acceptance_probability.png)


# Actually drawing from posterior

Green is prior. Red is posterior. Star is true value.

![Posterior distribution](https://raw.githubusercontent.com/Vlad-Shcherbina/bayesian_data_analysis/master/ch10/posterior.png)
