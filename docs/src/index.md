# FeedbackParticleFilters.jl

A package for using feedback particle filters in nonlinear stochastic filtering and data assimilation problems.

## What are feedback particle filters?

Feedback particle filters (FPFs) are a class of [sample-based numerical algorithms](https://ieeexplore.ieee.org/document/6530707) to approximate the conditional distribution in a nonlinear filtering problem.
In contrast to conventional particle filters, which use importance weights, FPFs use feedback control to let the observations guide the particles to the appropriate position.

## Features

This package's aim is to provide a versatile and efficient feedback particle filter implementation in Julia, with abstractions to flexibly construct, run, and analyze feedback particle filters for a variety of uni- and multivariate filtering problems with both diffusion and point process observations.

In particular, the following features are planned to be implemented in FeedbackParticleFilters:
* Types for hidden state and observation models: diffusions, Poisson processes, etc.
* A variety of gain estimation methods
* Automatic filter deployment and simulation of the state and filtering equations
* Storing of intermediate (trajectory) data from simulation
* An interface to the powerful solvers from the [DifferentialEquations](https://github.com/JuliaDiffEq/DifferentialEquations.jl) package 

## Manual outline

```@contents
Pages = [
 "getting_started.md",
 "filtering.md",
 "fpf.md",
 "hidden.md",
 "observation.md",
 "gainest.md",
 "reference.md",
]
Depth = 1
```
