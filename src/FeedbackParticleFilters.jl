module FeedbackParticleFilters

import StatsBase
import LinearAlgebra
import Distributions
#using Distributed
#using PyPlot

include("ParticleEnsembles.jl")
export
    UnweightedParticleEnsemble,
    FPFEnsemble

include("StateModels.jl")
export
    StateModel,
    DiffusionStateModel,
    ScalarDiffusionStateModel

include("ObservationModels.jl")
export
    ObservationModel,
    DiffusionObservationModel,
    ScalarDiffusionObservationModel,
    PointprocessObservationModel,
    ScalarPointprocessObservationModel

include("FilteringProblems.jl")
export
    FilteringProblem,
    ContinuousTimeFilteringProblem

include("GainEstimation.jl")
export
    GainEquation,
        PoissonEquation,
            ScalarPoissonEquation,
            ScalarVectorPoissonEquation,
            VectorScalarPoissonEquation,
            VectorPoissonEquation,
    Update!,
    Solve!,
    GainEstimationMethod,
        SemigroupMethod1d,
        GainDataSemigroup1d

include("Propagation.jl")
export
    ApplyGain!


end # module