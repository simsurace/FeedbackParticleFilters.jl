"""
    ParticleRepresentation{S} <: AbstractFilterRepresentation{S}

Abstract type for the representation of the conditional distribution over the hidden state by particles (samples), weighted or unweighted.
"""
abstract type ParticleRepresentation{S} <: AbstractFilterRepresentation{S} end



"""
    get_pos(ensemble::ParticleRepresentation, i)

Return a view of the position of the particle with index `i` from `ensemble`.
"""
function get_pos(ensemble::ParticleRepresentation, i) end





"""
    list_of_particles(ensemble::ParticleRepresentation)

Return a list of particle position views from `ensemble`.
"""
list_of_pos(ens::ParticleRepresentation) = [get_particle(ens, i) for i in 1:no_of_particles(ens)]




"""
    no_of_particles(ensemble::ParticleRepresentation)

Return the number of particles in `ensemble`.
"""
function no_of_particles(ensemble::ParticleRepresentation) end




"""
    eff_no_of_particles(ensemble::ParticleRepresentation)

Return the effective number of particles in `ensemble`. For unweighted ensembles, this is equal to the number of particles.
For weighted ensembles, this returns the inverse of the sum of the squared weights
"""
eff_no_of_particles(ensemble::ParticleRepresentation) = no_of_particles(ensemble)




"""
    particle_dim(ensemble::ParticleRepresentation)

Return the dimension of individual particles in `ensemble`.
"""
function particle_dim(ensemble::ParticleRepresentation) end




dim(ensemble::ParticleRepresentation) = particle_dim(ensemble) * no_of_particles(ensemble)









"""
    eltype(ensemble::ParticleRepresentation)

Return the type of individual particles in `ensemble`.
"""
Base.eltype(ensemble::ParticleRepresentation{S}) where S = S



#function propagate!(ens::ParticleRepresentation{S}, model::HiddenStateModel{S}) where S end
#function propagate!(ens::ParticleRepresentation{S}, model::ContinuousTimeHiddenStateModel{S}, dt) where S end




# How to add new particle representations:
# * Add a struct which is a subtype of ParticleRepresentation{S}, where S is the type to be represented
# * Implement a method for dim which returns the dimensionality of the representation.
# * Implement methods for
#   - get_pos
#   - no_of_particles
#   - particle_dim