"""
    ParticleFlowMethod

Abstract type for a method used to flow particles according to a prescribed change of the density.
"""
abstract type ParticleFlowMethod end




"""
    flow!(ensemble::UnweightedParticleEnsemble, method::ParticleFlowMethod) --> eq

Performs a particle flow of `ensemble` according to particle flow method `method`.
"""
function flow!(ensemble::UnweightedParticleRepresentation, method::ParticleFlowMethod) end