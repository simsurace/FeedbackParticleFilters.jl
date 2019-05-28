module FeedbackParticleFilters

using StatsBase
using LinearAlgebra
#using Distributed
#using PyPlot

include("ParticleEnsembles.jl")
export
    UnweightedParticleEnsemble,
    FPFEnsemble

include("ObservationModels.jl")
export
    ObservationModel,
    DiffusionObservationModel1d,
    ScalarObservationData,
    PointprocessObservationModel

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