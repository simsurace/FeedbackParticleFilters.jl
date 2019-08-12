abstract type GainEquation end
abstract type GainEstimationMethod end








"""
    solve!(eq::GainEquation, method::GainEstimationMethod) --> eq

Solves the gain equation `eq` using method `method`.
"""
function solve!(eq::GainEquation, method::GainEstimationMethod) end




"""
    update!(eq::GainEquation)

Updates the gain equation `eq`.

    update!(eq::GainEquation, ens::ParticleRepresentation)

Updates the gain equation `eq` by incorporating new information from the ensemble `ens`.
"""
function update!(eq::GainEquation) end
function update!(eq::GainEquation, ens::ParticleRepresentation) end




function state_dim(eq::GainEquation) end
function obs_dim(eq::GainEquation) end