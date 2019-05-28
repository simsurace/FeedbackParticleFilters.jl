module PFlib
# Particle filtering library for Julia 1.1.1
# May 2019
# (c) Simone Carlo Surace

export ParticleEnsemble,UnweightedParticleEnsemble,WeightedParticleEnsemble
export ApplyGain!,Gain_semigroup!

#using Pkg
using StatsBase
using LinearAlgebra
using Distributed
using PyPlot

# define a data type for particle filtering
# T is the type for the state space of the particles
abstract type ParticleEnsemble{T} end;

mutable struct UnweightedParticleEnsemble{T} <: ParticleEnsemble{T}
    positions::Array{T,1}
    size::Int64
    gain::Array{T,1}
    potential::Array{Float64,1}
end;

mutable struct WeightedParticleEnsemble{T} <: ParticleEnsemble{T}
    positions::Array{T,1}
    weights::ProbabilityWeights # built-in type that stores the sum of weights as weights.sum
    size::Int64
end;


#------------------------------------------------------------------------------
# Gain estimation methods for feedback particle filters
#------------------------------------------------------------------------------

# This function applies the gain stored in the ensemble object
function ApplyGain!(ensemble::UnweightedParticleEnsemble, dt::Float64)
   broadcast!(+, ensemble.positions, ensemble.positions, dt .* ensemble.gain)
end;

# Semigroup method from Taghvaei & Mehta, IEEE CDC, 2016
function Gain_semigroup!(ensemble::UnweightedParticleEnsemble, h::Function)
    epsilon = .1
    N = ensemble.size
    H = zeros(Float64, N)
    broadcast!(h, H, ensemble.positions)
    broadcast!(-, H, H, mean(H))
    broadcast!(*, H, H, epsilon)

    # compute T operator
    T = zeros(Float64, N, N)
    for i in 1:N
        for j in i:N
            T[i,j] = exp(-(ensemble.positions[i]-ensemble.positions[j])^2/(4*epsilon))
            T[j,i] = T[i,j]
        end
    end
    broadcast!(/, T, T, sqrt.(sum(T,dims=1) .* sum(T,dims=2)))
    broadcast!(/, T, T, sum(T,dims=2))

    # solve fixed-point equation
    newpotential = copy(ensemble.potential)::Array{Float64,1}
    fluctuation = 1.
    while fluctuation > 1E-2
        mul!(newpotential, T, ensemble.potential)
        broadcast!(+, newpotential, newpotential, H)
        broadcast!(-, newpotential, newpotential, mean(newpotential))
        fluctuation = maximum(abs.(newpotential-ensemble.potential))
        ensemble.potential = copy(newpotential)
    end

    ensemble.gain = T * (ensemble.potential .* ensemble.positions) - (T*ensemble.potential) .* (T*ensemble.positions)
    broadcast!(/, ensemble.gain, ensemble.gain, 2*epsilon)
end;


function Gain_semigroup!(ensemble::UnweightedParticleEnsemble) # For h(x)=x
    epsilon = .1
    N = ensemble.size
    H = copy(ensemble.positions)
    broadcast!(-, H, H, mean(H))
    broadcast!(*, H, H, epsilon)

    # compute T operator
    T = zeros(Float64, N, N)
    for i in 1:N
        for j in i:N
            T[i,j] = exp(-(ensemble.positions[i]-ensemble.positions[j])^2/(4*epsilon))
            T[j,i] = T[i,j]
        end
    end
    broadcast!(/, T, T, sqrt.(sum(T,dims=1) .* sum(T,dims=2)))
    broadcast!(/, T, T, sum(T,dims=2))

    # solve fixed-point equation
    newpotential = copy(ensemble.potential)::Array{Float64,1}
    fluctuation = 1.
    while fluctuation > 1E-2
        mul!(newpotential, T, ensemble.potential)
        broadcast!(+, newpotential, newpotential, H)
        broadcast!(-, newpotential, newpotential, mean(newpotential))
        fluctuation = maximum(abs.(newpotential-ensemble.potential))
        ensemble.potential = copy(newpotential)
    end

    ensemble.gain = T * (ensemble.potential .* ensemble.positions) - (T*ensemble.potential) .* (T*ensemble.positions)
    broadcast!(/, ensemble.gain, ensemble.gain, 2*epsilon)
end;

end
