#!/usr/bin/env julia

using FeedbackParticleFilters, Test, SafeTestsets

println("Starting tests")
@safetestset "Particle ensembles" begin include("test_ParticleEnsembles.jl") end
@safetestset "State models" begin include("test_StateModels.jl") end
@safetestset "Observation models" begin include("test_ObservationModels.jl") end
@safetestset "Filtering problems" begin include("test_FilteringProblems.jl") end
@safetestset "Gain equations" begin include("test_GainEquations.jl") end
@safetestset "Gain estimation" begin include("test_GainEstimation.jl") end
@safetestset "Gain estimation methods" begin include("test_GainEstimationMethods.jl") end