# Reference

**Contents**

```@meta
CurrentModule = FeedbackParticleFilters
```

```@contents
Pages = ["reference.md"]
```

## Basic abstractions

```@docs
AbstractHiddenState
VectorHiddenState
AbstractFilterRepresentation
ParticleRepresentation
UnweightedParticleRepresentation
AbstractGainEquation
EmptyGainEquation
```

## Basic methods

```@docs
eltype
Map
```

## Hidden state models

```@docs
StateModel
DiffusionStateModel
ScalarDiffusionStateModel
Propagator
```