# Hidden state models

```@meta
CurrentModule = FeedbackParticleFilters
```

```@docs
HiddenStateModel
state_dim
state_type
time_type
initial_condition
initialize
propagate!
```

```@contents
Pages = ["hidden.md"]
```

## Diffusion processes

```@docs
DiffusionStateModel
LinearDiffusionStateModel
ScalarDiffusionStateModel
drift_function
diffusion_function
noise_dim
```