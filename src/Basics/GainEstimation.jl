"""
    GainEquation

Abstract type for a container representing a gain (vector field).
"""
abstract type GainEquation end




"""
    GainEstimationMethod

Abstract type for a method used to solve an equation of `GainEquation` type.
"""
abstract type GainEstimationMethod end




"""
    solve!(eq::GainEquation, method::GainEstimationMethod) --> eq

Solves the gain equation `eq` using method `method`.
"""
function solve!(eq::GainEquation, method::GainEstimationMethod) end




"""
    update!(eq::GainEquation)

Updates the gain equation `eq` such that all information contained in it is self-consistent.

    update!(eq::GainEquation, ens::ParticleRepresentation)

Updates the gain equation `eq` by incorporating new information from the ensemble `ens`.
"""
function update!(eq::GainEquation) end
function update!(eq::GainEquation, ens::ParticleRepresentation) end




function state_dim(eq::GainEquation) end
function obs_dim(eq::GainEquation) end