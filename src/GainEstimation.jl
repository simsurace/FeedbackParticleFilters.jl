################################
### Gain equation constructs ###
################################

abstract type GainEquation end
function GainEquation(state_model::StateModel, obs_model::ObservationModel) end
function GainEquation(state_model::StateModel, obs_model::ObservationModel, ensemble::FPFEnsemble) end

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

function GainEquation(state_model::ScalarDiffusionStateModel, obs_model::ScalarDiffusionObservationModel, N::Int)
    ScalarPoissonEquation(obs_model.observation_function, N)
end

function GainEquation(state_model::ScalarDiffusionStateModel, obs_model::ScalarDiffusionObservationModel, ensemble::FPFEnsemble)
    ScalarPoissonEquation(obs_model.observation_function, ensemble)
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





#######################
### Gain estimation ###
#######################

"""
    GainEstimationMethod

Any type of gain estimation method for the feedback particle filter. 
Implemented abstract methods (consult individual documentation for concrete methods):

SemigroupMethod
"""
abstract type GainEstimationMethod end


"""
    Solve!(eq::GainEquation, method::GainEstimationMethod)

Solve the gain equation `eq` using method `method`.
"""
function Solve!(eq::GainEquation, method::GainEstimationMethod) end

"""
    Update!(eq::GainEquation, ensemble::FPFEnsemble)

Update the gain equation `eq` using data from particle ensemble `ensemble`, i.e. compute observation function at particle positions.
"""
function Update!(eq::GainEquation, ensemble::FPFEnsemble) end

function Update!(eq::ScalarPoissonEquation, ensemble::FPFEnsemble)
    eq.positions = ensemble.positions
    broadcast!(eq.h, eq.H, eq.positions)
    eq.mean_H = StatsBase.mean(eq.H)
end

function Update!(eq::VectorScalarPoissonEquation, ensemble::FPFEnsemble)
    eq.positions = ensemble.positions
    broadcast!(eq.h, eq.H, eq.positions)
    eq.mean_H = StatsBase.mean(eq.H)
end


########################
### Semigroup method ###
########################

"""
    SemigroupMethod

Semigroup method from Algorithm 1 in [1].

Concrete methods: SemigroupMethod1d

[1] Taghvaei, A., & Mehta, P. G. (2016). Gain function approximation in the feedback particle filter. In 2016 IEEE 55th Conference on Decision and Control (CDC) (pp. 5446–5452). IEEE. https://doi.org/10.1109/CDC.2016.7799105
"""
abstract type SemigroupMethod <: GainEstimationMethod end

"""
    SemigroupMethod1d(epsilon, delta)

One-dimensional semigroup method with Gaussian kernels of variance epsilon. 
The fixed-point equation is iterated as long as the maximum change in the potential is larger than delta.
"""
struct SemigroupMethod1d <: SemigroupMethod
    epsilon::Float64
    delta::Float64
end

function Solve!(eq::ScalarPoissonEquation, method::SemigroupMethod1d) 
    N = length(eq.positions)
    H = copy(eq.H)
    broadcast!(-, H, H, eq.mean_H)   
    broadcast!(*, H, H, method.epsilon)

    # compute T operator
    T = zeros(Float64, N, N)
    for i in 1:N
        for j in i:N
            T[i,j] = exp(-(eq.positions[i]-eq.positions[j])^2/(4*method.epsilon))
            T[j,i] = T[i,j]
        end
    end
    broadcast!(/, T, T, sqrt.(sum(T,dims=1) .* sum(T,dims=2)))
    broadcast!(/, T, T, sum(T,dims=2))

    # solve fixed-point equation
    newpotential = copy(eq.potential)::Array{Float64,1}
    fluctuation = 1.
    while fluctuation > method.delta
        LinearAlgebra.mul!(newpotential, T, eq.potential)
        broadcast!(+, newpotential, newpotential, H)
        broadcast!(-, newpotential, newpotential, StatsBase.mean(newpotential))
        fluctuation = maximum(abs.(newpotential-eq.potential))
        eq.potential = copy(newpotential)
    end

    eq.gain = T * (eq.potential .* eq.positions) - (T*eq.potential) .* (T*eq.positions)
    broadcast!(/, eq.gain, eq.gain, 2*method.epsilon)
end


