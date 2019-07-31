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



"""
    AbstractModel{T}

Abstract type for any model over states of type `T`.
"""
abstract type AbstractModel{T} end




"""
    HiddenStateModel{T} <: AbstractModel{T}

Abstract type for any model of the hidden state of type `T`.
"""
abstract type HiddenStateModel{T} <: AbstractModel{T} end




"""
    ObservationModel{S, T} <: AbstractModel{T}

Abstract type for any model of observations of type `T` and hidden states of type `S`.
"""
abstract type ObservationModel{S, T} <: AbstractModel{T} end




"""
    ContinuousTimeHiddenStateModel{T} <: HiddenStateModel{T} <: AbstractModel{T}

Abstract type for any continuous-time model of the hidden state of type `T`.
"""
abstract type ContinuousTimeHiddenStateModel{T} <: HiddenStateModel{T} end




"""
    ContinuousTimeObservationModel{S, T} <: ObservationModel{S, T} <: AbstractModel{T}

Abstract type for any continuous-time model of the observations of type `T` and hidden states of type `S`.
"""
abstract type ContinuousTimeObservationModel{S, T} <: ObservationModel{S, T} end




"""
    AbstractProblem{S, T}

Abstract type for a problem for observations of type `T` and hidden states of type `S`.
"""
abstract type AbstractProblem{S, T, M1<:HiddenStateModel, M2<:ObservationModel} end




"""
    AbstractFilteringProblem{S, T, M1<:HiddenStateModel{S}, M2<:ObservationModel{S,T}}

Abstract type for a filtering problem for observations of type `T` and hidden states of type `S`.
"""
const AbstractFilteringProblem = AbstractProblem{S, T, <:HiddenStateModel{S}, <:ObservationModel{S, T}} where {S,T}

function Base.show(io::IO, problem::AbstractFilteringProblem{S, T}) where {S,T}
    println(io, "Generic filtering problem")
    println(io, "    Type of hidden state:     ", S)
    println(io, "    Type of observation:      ", T)
    println(io, "    Hidden state model:       ", problem.state_model)
    println(io, "    Observation model:        ", problem.obs_model)
end






"""
    ContinuousTimeFilteringProblem{S, T, M1<:ContinuousTimeHiddenStateModel{S}, M2<:ContinuousTimeObservationModel{S,T}}

Abstract type for a filtering problem in continuous time for observations of type `T` and hidden states of type `S`.
"""
const ContinuousTimeFilteringProblem = AbstractFilteringProblem{S, T, <:ContinuousTimeHiddenStateModel{S}, <:ContinuousTimeObservationModel{S, T}} where {S,T}

function Base.show(io::IO, problem::ContinuousTimeFilteringProblem{S, T}) where {S,T}
    println(io, "Continuous-time filtering problem")
    println(io, "    Type of hidden state:     ", S)
    println(io, "    Type of observation:      ", T)
    println(io, "    Hidden state model:       ", problem.state_model)
    println(io, "    Observation model:        ", problem.obs_model)
end






"""
    AbstractFilterRepresentation{S}

Abstract type for representation of the conditional distribution over the hidden state of type `S`.
"""
abstract type AbstractFilterRepresentation{S} end;




"""
    ParticleRepresentation{S} <: AbstractFilterRepresentation{S}

Abstract type for the representation of the conditional distribution over the hidden state by particles (samples), weighted or unweighted.
"""
abstract type ParticleRepresentation{S} <: AbstractFilterRepresentation{S} end




"""
    UnweightedParticleRepresentation{S} <: ParticleRepresentation{S} <: AbstractFilterRepresentation{S}

Abstract type for the representation of the conditional distribution over the hidden state by unweighted particles (samples).
"""
abstract type UnweightedParticleRepresentation{S} <: ParticleRepresentation{S} end




"""
    UnweightedParticleRepresentation{S} <: ParticleRepresentation{S} <: AbstractFilterRepresentation{S}

Abstract type for the representation of the conditional distribution over the hidden state by unweighted particles (samples).
"""
#abstract type AbstractFilter{P<:AbstractFilteringProblem{S, T}, R<:AbstractFilterRepresentation{S}} where {S, T} end




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