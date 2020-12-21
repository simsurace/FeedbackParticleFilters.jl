#!/usr/bin/env julia

using FeedbackParticleFilters, Test, SafeTestsets


println("Starting tests")


@safetestset "Basics" begin
    println("")
    println("")
    println("")
    printstyled("Basics", bold=true)
    println("")
    include("Basics/test_Abstractions.jl")
    include("Basics/test_AbstractModel.jl")
    include("Basics/test_AbstractFilteringProblem.jl")
    include("Basics/test_AbstractFilterRepresentation.jl")
    include("Basics/test_AbstractFilteringAlgorithm.jl")
    include("Basics/test_HiddenStateModel.jl")
    include("Basics/test_ObservationModel.jl")
    include("Basics/test_ParametricRepresentation.jl")
    include("Basics/test_ParticleRepresentation.jl")
    include("Basics/test_UnweightedParticleRepresentation.jl")
    include("Basics/test_WeightedParticleRepresentation.jl")
    include("Basics/test_GainEstimation.jl")
    include("Basics/test_FilteringProblem.jl")
end


@safetestset "State models" begin
    println("")
    println("")
    println("")
    printstyled("State models", bold=true)
    println("")
    include("StateModels/test_DiffusionStateModel.jl")
    include("StateModels/test_LinearDiffusionStateModel.jl")
end


@safetestset "Observation models" begin
    println("")
    println("")
    println("")
    printstyled("Observation models", bold=true)
    println("")
    include("ObservationModels/test_DiffusionObservationModel.jl")
    include("ObservationModels/test_LinearDiffusionObservationModel.jl")
end


@safetestset "Filter representations" begin
    println("")
    println("")
    println("")
    printstyled("Filter representations", bold=true)
    println("")
    include("FilterRepresentations/test_MultivariateGaussian.jl")
    include("FilterRepresentations/test_UnweightedParticleEnsemble.jl")
    include("FilterRepresentations/test_WeightedParticleEnsemble.jl")
end


@safetestset "Gain equations" begin 
    println("")
    println("")
    println("")
    printstyled("Gain equations", bold=true)
    println("")
    include("GainEquations/test_PoissonEquation.jl")
end


@safetestset "Gain estimation methods" begin 
    println("")
    println("")
    println("")
    printstyled("Gain estimation methods", bold=true)
    println("")
    include("GainEstimationMethods/test_ConstantGainApproximation.jl")
    include("GainEstimationMethods/test_ConstantGainEKSPF.jl")
    include("GainEstimationMethods/test_SemigroupMethod.jl")
    include("GainEstimationMethods/test_DifferentialRKHSMethod.jl")
    include("GainEstimationMethods/test_DifferentialRKHSMethodS1.jl")
end


@safetestset "Filtering algorithms" begin 
    println("")
    println("")
    println("")
    printstyled("Filtering algorithms", bold=true)
    println("")
    include("FilteringAlgorithms/test_BPF.jl")
    include("FilteringAlgorithms/test_KBF.jl")
    include("FilteringAlgorithms/test_FPF.jl")
end


@safetestset "Simulation" begin 
    println("")
    println("")
    println("")
    printstyled("Simulation", bold=true)
    println("")
    include("Simulation/test_Simulation.jl")
    include("Simulation/test_SimulationState.jl")
end

