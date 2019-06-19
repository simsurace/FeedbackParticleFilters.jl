################################
### Gain equation constructs ###
################################

abstract type GainEquation end

abstract type PoissonEquation <: GainEquation end 

"""
    ScalarPoissonEquation

A scalar Poisson equation, i.e. the right-hand side and the potential are both scalar.

    ScalarPoissonEquation(h::Function, N::Int)

Initialize a scalar Poisson equation for N particles.

    ScalarPoissonEquation(h::Function, ensemble::FPFEnsemble)

Initialize a scalar Poisson equation for the given ensemble.
"""
mutable struct ScalarPoissonEquation <: PoissonEquation
    h::Function
    positions::Array{Float64,1}
    H::Array{Float64,1}
    mean_H::Float64
    potential::Array{Float64,1}
    gain::Array{Float64,1}
    ScalarPoissonEquation(h, positions, H, mean_H, potential, gain) = 
        if length(positions) == length(H) == length(potential) == length(gain)
            new(h, positions, H, mean_H, potential, gain)
        else
            error("ERROR: length mismatch")
        end#if
end
    
function ScalarPoissonEquation(h::Function, N::Int)
    ScalarPoissonEquation(x->x, randn(N), zeros(Float64,N), 0., ones(Float64,N), zeros(Float64,N))
end
    
function ScalarPoissonEquation(h::Function, ensemble::FPFEnsemble)
    H = h.(ensemble.positions)
    N = length(H)
    ScalarPoissonEquation(h, ensemble.positions, H, StatsBase.mean(H), ones(Float64,N), zeros(Float64,N))
end

mutable struct VectorScalarPoissonEquation <: PoissonEquation
    h::Function
    positions::Array{Float64,2}
    H::Array{Float64,1}
    mean_H::Float64
    potential::Array{Float64,1}
    gain::Array{Float64,2}
end
function VectorScalarPoissonEquation(h::Function, ensemble::FPFEnsemble)
    H = h.(ensemble.positions)
    N = length(H)
    VectorScalarPoissonEquation(h, ensemble.positions, H, StatsBase.mean(H), ones(Float64,N), zeros(Float64,N,N))
end

mutable struct ScalarVectorPoissonEquation <: PoissonEquation
    h::Function
    positions::Array{Float64,1}
    H::Array{Float64,2}
    mean_H::Array{Float64,1}
    potential::Array{Float64,2}
    gain::Array{Float64,2}
end

mutable struct VectorPoissonEquation <: PoissonEquation
    h::Function
    positions::Array{Float64,2}
    H::Array{Float64,2}
    mean_H::Array{Float64,1}
    potential::Array{Float64,2}
    gain::Array{Float64,3}
end


##################################
### Gain equations from models ###
##################################
    
function GainEquation(filtering_problem::ContinuousTimeFilteringProblem)
        GainEquation(filtering_problem.state_model, filtering_problem.obs_model)
end
    
function GainEquation(filtering_problem::ContinuousTimeFilteringProblem, ensemble::FPFEnsemble)
        GainEquation(filtering_problem.state_model, filtering_problem.obs_model, ensemble)
end
    
"""
    GainEquation(state_model::HiddenStateModel, obs_model::ObservationModel, N::Int)

This will automatically construct a GainEquation object for the specified models and `N` particles.
"""
function GainEquation(state_model::HiddenStateModel, obs_model::ObservationModel, N::Int) end

"""
    GainEquation(state_model::StateModel, obs_model::ObservationModel, ensemble::FPFEnsemble)

This will automatically construct a GainEquation object for the specified models and the concrete ensemble.
"""
function GainEquation(state_model::HiddenStateModel, obs_model::ObservationModel, ensemble::FPFEnsemble) end
    
function GainEquation(state_model::ScalarDiffusionStateModel, obs_model::ScalarDiffusionObservationModel, N::Int)
    ScalarPoissonEquation(obs_model.observation_function, N)
end

function GainEquation(state_model::ScalarDiffusionStateModel, obs_model::ScalarDiffusionObservationModel, ensemble::FPFEnsemble)
    ScalarPoissonEquation(obs_model.observation_function, ensemble)
end