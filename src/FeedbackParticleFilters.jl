module FeedbackParticleFilters

import StatsBase
import LinearAlgebra
import Distributions
#using Distributed
#using PyPlot

include("BasicAbstractions.jl")
export
    AbstractHiddenState,
    VectorHiddenState,
    AbstractModel,
    HiddenStateModel,
    ObservationModel,
    ContinuousTimeHiddenStateModel,
    ContinuousTimeObservationModel,
    AbstractFilteringProblem,
    AbstractFilterRepresentation,
    ParticleRepresentation,
    UnweightedParticleRepresentation,
    AbstractGainEquation,
    EmptyGainEquation

include("BasicMethods.jl")
export
    eltype,
    Map

include("ParticleEnsembles.jl")
export
    UnweightedParticleEnsemble,
    FPFEnsemble

include("StateModels.jl")
export
    DiffusionStateModel,
    ScalarDiffusionStateModel,
    Propagator

include("ObservationModels.jl")
export
    DiffusionObservationModel,
    ScalarDiffusionObservationModel,
    PointprocessObservationModel,
    ScalarPointprocessObservationModel,
    Emitter

include("FilteringProblems.jl")
export
    FilteringProblem,
    ContinuousTimeFilteringProblem

include("GainEquations.jl")
export
    GainEquation,
        PoissonEquation,
            ScalarPoissonEquation,
            ScalarVectorPoissonEquation,
            VectorScalarPoissonEquation,
            VectorPoissonEquation

include("GainEstimation.jl")
export
    Update!,
    Solve!

include("GainEstimationMethods.jl")
export
    GainEstimationMethod,
        SemigroupMethod1d,
        RegularizedSemigroupMethod1d

include("Propagation.jl")
export
    ApplyGain!

include("Filters.jl")
export
    Filter,
    ContinuousTimeFilter,
    FeedbackParticleFilter,
    Simulation,
    FPFSimulation,
    run!

include("Utilities.jl")
export
    MSE,
    RelativeMSE


end # module