# FeedbackParticleFilters.jl
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://simsurace.github.io/FeedbackParticleFilters.jl/dev)
[![ci](https://github.com//simsurace/FeedbackParticleFilters.jl/actions/workflows/ci.yml/badge.svg)](https://github.com//simsurace/FeedbackParticleFilters.jl/actions/workflows/ci.yaml)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![codecov](https://codecov.io/gh/simsurace/FeedbackParticleFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/simsurace/FeedbackParticleFilters.jl)



This package's aim is to provide a versatile and efficient feedback particle filter implementation in Julia, with abstractions to flexibly construct, run, and analyze feedback particle filters for a variety of uni- and multivariate filtering problems with both diffusion and point process observations.

It provides implementations of the following algorithms:
- Kalman-Bucy filter `KBF`
- Feedback Particle Filter `FPF`
- Bootstrap Particle Filter (weighted) `BPF`
- Point-process Feedback Particle Filter `ppFPF`
- Ensemble Kushner-Stratonovich-Poisson Filter `EKSPF`

as well as
* Hidden state and observation models: diffusions, Poisson processes, etc.
* A variety of gain estimation methods: constant gain, semigroup, reproducing kernel Hilbert space, etc.
* Deterministic particle flow

If you have questions or comments, please open an issue!

## Installation

This package is officially registered.
To install it, use the built-in package manager:

```julia
pkg> add FeedbackParticleFilters
```
The package is currently tested on Julia 1.6-1.8, but should work on earlier versions too.

## Usage

To load the package, use the command:
```julia
using FeedbackParticleFilters
```
Set up a basic one-dimensional linear-Gaussian continuous-time filtering problem:
```julia
using Distributions
state_model = ScalarDiffusionStateModel(x->-x, x->sqrt(2.), Normal())
obs_model   = ScalarDiffusionObservationModel(x->x)

filt_prob   = FilteringProblem(state_model, obs_model)
```
Once the filtering problem is defined, an appropriate filtering algorithm can be defined like this:
```julia
method = ConstantGainApproximation()
filter = FPF(filt_prob, method, 100)
```
The package comes with methods to automatically simulate a given system:
```julia
simulation = ContinuousTimeSimulation(filt_prob, filter, 10000, 0.01)
run!(simulation)
```
To learn more about how to use this package, please check out some tutorials or the documentation linked below.

## Tutorials

There are various Jupyter notebooks that explore various key functions of the package:
1. [Getting started](https://github.com/simsurace/FeedbackParticleFilters.jl/blob/master/notebooks/Getting_started.ipynb)
2. Gain estimation using the [semigroup method](https://github.com/simsurace/FeedbackParticleFilters.jl/blob/master/notebooks/Gain_semigroup.ipynb)
3. [Harmonic oscillator example](https://github.com/simsurace/FeedbackParticleFilters.jl/blob/master/notebooks/Harmonic_oscillator.ipynb)

## Documentation

[In development...](https://simsurace.github.io/FeedbackParticleFilters.jl/dev)

## Acknowledgements

This package was developed as part of academic research at Department of Physiology, University of Bern, Switzerland.
The research was funded by the Swiss National Science Foundation.
