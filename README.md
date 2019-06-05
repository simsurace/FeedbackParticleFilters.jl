# FeedbackParticleFilters.jl
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://simsurace.github.io/FeedbackParticleFilters.jl/latest)
[![Build Status](https://travis-ci.org/simsurace/FeedbackParticleFilters.jl.svg?branch=master)](https://travis-ci.org/simsurace/FeedbackParticleFilters.jl)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![codecov](https://codecov.io/gh/simsurace/FeedbackParticleFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/simsurace/FeedbackParticleFilters.jl)



This package provides a versatile and efficient feedback particle filter implementation in Julia, with abstractions to flexibly construct, run, and analyze feedback particle filters for a variety of uni- and multivariate filtering problems with both diffusion and point process observations.

In particular, FeedbackParticleFilters implements:
* Types for hidden state and observation models: diffusions, Poisson processes, etc.
* A variety of gain estimation methods
* Automatic filter deployment and simulation of the state and filtering equations
* Storing of intermediate (trajectory) data from simulation
* An interface to the powerful solvers from the [DifferentialEquations](https://github.com/JuliaDiffEq/DifferentialEquations.jl) package 

## Installation

In Julia:

```julia
Pkg.add("FeedbackParticleFilters")
```

## Usage

To load the package, use the command:
```julia
using FeedbackParticleFilters
using Distributions
```
Set up a basic one-dimensional linear-Gaussian continuous-time filtering problem:
```
state_model = ScalarDiffusionStateModel(x->-x, x->sqrt(2.), Normal())
obs_model = ScalarDiffusionObservationModel(x->x, x->1)

filt_prob = ContinuousTimeFilteringProblem(state_model, obs_model)
```
Once the filtering problem is defined, you can use it to perform a variety of tasks.

For example, you may initialize an ensemble of `N=100` particles:
```
ensemble = FPFEnsemble(state_model, 100)
```
The following generates a Poisson equation for the gain using the ensemble above.
The equation is solved using the semigroup gain estimation method.
```
eq = GainEquation(state_model, obs_model, ensemble)
method = SemigroupMethod1d(1E-1,1E-2)
Solve!(eq, method)
```
The gain at the particle locations is stored in `eq.gain`.
These low-level building blocks can then be used to write custom numerical implementations.
The package also comes with methods to automatically simulate given filtering problem:
```
filter = FeedbackParticleFilter(filt_prob, method, 100);
simulation = FPFSimulation(filter, 10000, 0.01);
run!(simulation)
```
To learn more about how to use this package, please check out some tutorials or the documentation linked below.

## Tutorials

There are various Jupyter notebooks that explore various key functions of the package:
1. [Getting started](notebooks/Getting_started.ipynb)
2. Gain estimation using the [semigroup method](notebooks/Gain_semigroup.ipynb)

## Documentation

https://simsurace.github.io/FeedbackParticleFilters.jl/latest
