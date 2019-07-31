@doc raw"""
    UnweightedParticleEnsemble{T}

An ensemble of `N` particles, each of dimension `n`.
"""
struct UnweightedParticleEnsemble{T} <: UnweightedParticleRepresentation{Vector{T}}
    positions::Matrix{T}
end


particle_dim(ens::UnweightedParticleEnsemble)    = size(ens.positions, 1)









no_of_particles(ens::UnweightedParticleEnsemble) = size(ens.positions, 2)









get_pos(ens::UnweightedParticleEnsemble{T}, i) where T<:AbstractMatrix = view(ens.positions, :, i)









Base.show(io::IO, ens::UnweightedParticleEnsemble) = print(io, "Unweighted particle ensemble
    # of particles: ", no_of_particles(ens),"
    particle type:  ", particle_dim(ens)"-dimensional ", eltype(ens))







function propagate!(ens::UnweightedParticleEnsemble{S}, model::HiddenStateModel{S}, dt)
    propagate!(ens.positions, model, dt)
end