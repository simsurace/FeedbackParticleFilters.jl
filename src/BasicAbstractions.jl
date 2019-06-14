#############
### Types ###
#############


"""
    AbstractHiddenState = Union{Number, AbstractVector{T}

Abstract type for the state space of the hidden state.
"""
AbstractHiddenState = Union{Number, AbstractVector{T} where T<:Number}

"""
    VectorHiddenState = Union{Float64, AbstractVector{T} where T<:Float64}

Abstract type for those state spaces of the hidden state that shall be treated as vector spaces (i.e. support addition and multiplication by scalars). 
This excludes points from a manifold, for example.
"""
VectorHiddenState = Union{Float64, AbstractVector{T} where T<:Float64}

#abstract type AbstractParameter <: Union{Number, AbstractVector{T} where T<:Number} end;

"""
    AbstractFilterRepresentation{T<:AbstractHiddenState}

Abstract type for representation of the conditional distribution over the hidden state.
"""
abstract type AbstractFilterRepresentation{T<:AbstractHiddenState} end;

#abstract type ParametricFilterRepresentation{T<:AbstractHiddenState, S<:AbstractParameter, P<:Union{S, AbstractVector{S}}} <: AbstractFilterRepresentation end

"""
    ParticleRepresentation{T<:AbstractHiddenState}

Abstract type for the representation of the conditional distribution over the hidden state by particles (samples), weighted or unweighted.
"""
abstract type ParticleRepresentation{T<:AbstractHiddenState} <: AbstractFilterRepresentation{T} end

"""
    UnweightedParticleRepresentation{T<:AbstractHiddenState}

Abstract type for the representation of the conditional distribution over the hidden state by unweighted particles (samples).
"""
abstract type UnweightedParticleRepresentation{T<:AbstractHiddenState} <: ParticleRepresentation{T} end

"""
    AbstractGainEquation{T<:AbstractHiddenState}

Abstract type for an equation that determines the gain (vector field).
"""
abstract type AbstractGainEquation{T<:AbstractHiddenState} end;

"""
    EmptyGainEquation{T<:AbstractHiddenState}

An empty gain equation, i.e. no information about the gain is known.
"""
struct EmptyGainEquation{T<:AbstractHiddenState} <: AbstractGainEquation{T} end;