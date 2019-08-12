# Getting started

## Installation

Use the built-in package manager:

```julia
using Pkg
Pkg.add("FeedbackParticleFilters")
```

## Basic usage

To load the package, use the command:
```julia
using FeedbackParticleFilters
```
Set up a basic one-dimensional linear-Gaussian continuous-time filtering problem:
```julia
using Distributions
state_model = ScalarDiffusionStateModel(x->-x, x->sqrt(2.), Normal())
obs_model   = ScalarDiffusionObservationModel(x->x)

filt_prob = ContinuousTimeFilteringProblem(state_model, obs_model)
```
Once the filtering problem is defined, you can use it to perform a variety of tasks.

For example, you may initialize an ensemble of `N=100` particles:
```julia
ensemble = UnweightedParticleEnsemble(state_model, 100)
```
The following generates a Poisson equation for the gain using the ensemble above.
The equation is solved using the semigroup gain estimation method.
```julia
eq = GainEquation(state_model, obs_model, ensemble)
method = SemigroupMethod(1E-1,1E-2)
solve!(eq, method)
```
The gain at the particle locations is stored in `eq.gain`.
These low-level building blocks can then be used to write custom numerical implementations.
The package also comes with methods to automatically simulate a given filtering problem:
```julia
filter = FPF(filt_prob, method, 100)
simulation = ContinuousTimeSimulation(filt_prob, filter, 10000, 0.01)
run!(simulation)
```


