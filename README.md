# FeedbackParticleFilters.jl
![Lifecycle](https://img.shields.io/badge/lifecycle-maturing-blue.svg)<!--
![Lifecycle](https://img.shields.io/badge/lifecycle-experimental-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-stable-green.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-retired-orange.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-archived-red.svg)
![Lifecycle](https://img.shields.io/badge/lifecycle-dormant-blue.svg) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://simsurace.github.io/FeedbackParticleFilters.jl/dev)
[![Build Status](https://travis-ci.org/simsurace/FeedbackParticleFilters.jl.svg?branch=master)](https://travis-ci.org/simsurace/FeedbackParticleFilters.jl)<!--
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)-->
[![codecov](https://codecov.io/gh/simsurace/FeedbackParticleFilters.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/simsurace/FeedbackParticleFilters.jl)



This package's aim is to provide a versatile and efficient feedback particle filter implementation in Julia, with abstractions to flexibly construct, run, and analyze feedback particle filters for a variety of uni- and multivariate filtering problems with both diffusion and point process observations.

In particular, the following features are planned to be implemented in FeedbackParticleFilters:
* Types for hidden state and observation models: diffusions, Poisson processes, etc.
* A variety of gain estimation methods
* Automatic filter deployment and simulation of the state and filtering equations
* Storing of intermediate (trajectory) data from simulation
* An interface to the powerful solvers from the [DifferentialEquations](https://github.com/JuliaDiffEq/DifferentialEquations.jl) package 

## Installation

Use the built-in package manager:

```julia
using Pkg
Pkg.add("FeedbackParticleFilters")
```

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
1. [Getting started](https://nbviewer.jupyter.org/github/simsurace/FeedbackParticleFilters.jl/blob/master/notebooks/Getting_started.ipynb)
2. Gain estimation using the [semigroup method](https://nbviewer.jupyter.org/github/simsurace/FeedbackParticleFilters.jl/blob/master/notebooks/Gain_semigroup.ipynb)
3. [Harmonic oscillator example](https://nbviewer.jupyter.org/github/simsurace/FeedbackParticleFilters.jl/blob/master/notebooks/Harmonic_oscillator.ipynb)

## Documentation

[In development...](https://simsurace.github.io/FeedbackParticleFilters.jl/dev)

## Acknowledgements

This package was developed as part of academic research at Department of Physiology, University of Bern, Switzerland.
The research was funded by the Swiss National Science Foundation.