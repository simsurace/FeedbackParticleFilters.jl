abstract type GainEquation end









"""
    solve!(eq::GainEquation, method::GainEstimationMethod) --> eq

Solves the gain equation `eq` using method `method`.
"""
function solve!(eq::GainEquation, method::GainEstimationMethod) end




"""
    update!(eq::GainEquation, ens::ParticleRepresentation)

Updates the gain equation `eq` by incorporating information from the ensemble `ens`.
"""
function update!(eq::GainEquation, ens::ParticleRepresentation) end