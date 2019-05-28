# FeedbackParticleFilters.jl
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://simsurace.github.io/FeedbackParticleFilters.jl/latest)

This package provides a versatile and efficient feedback particle filter implementation in Julia, with abstractions to flexibly construct, run, and analyze feedback particle filters for a variety of uni- and multivariate filtering problems with both diffusion and point process observations.

In particular, FeedbackParticleFilters implements:
* Types for hidden state and observation models: diffusions, Poisson processes, etc.
* A variety of gain estimation methods
* An interface to the powerful solvers from the `DifferentialEquations` package

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
method = SemigroupMethod1d(0.1,0.01)
Solve!(eq, method)
```
The gain at the particle locations is stored in `eq.gain`.
These low-level building blocks can then be used to write custom numerical implementations.
The package also interfaces with powerful solvers from the `DifferentialEquationsjl` package in order to simulate the system and the filter.
```
filter = FeedbackParticleFilter(filt_prob, method, 100)
using DifferentialEquations
trajectories = Simulate(filt_prob, filter, EM(), dt=0.01) # solve SDEs using the Euler-Maruyama method
```
## Tutorials

There are various Jupyter notebooks that explore various key functions of the package:
1. Bais [tutorial](notebooks/Basic_tutorial.ipynb)
2. Gain estimation using the [semigroup method](notebooks/Gain_semigroup.ipynb)

## Documentation

https://simsurace.github.io/FeedbackParticleFilters.jl/latest
