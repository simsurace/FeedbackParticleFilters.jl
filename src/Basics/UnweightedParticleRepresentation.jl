"""
    UnweightedParticleRepresentation{S} <: ParticleRepresentation{S} <: AbstractFilterRepresentation{S}

Abstract type for the representation of the conditional distribution over the hidden state by unweighted (i.e. equally weighted) particles (samples).
"""
abstract type UnweightedParticleRepresentation{S} <: ParticleRepresentation{S} end