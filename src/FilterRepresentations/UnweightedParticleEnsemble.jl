@doc raw"""
    UnweightedParticleEnsemble{T}

An ensemble of `N` particles, each of dimension `n`.
"""
struct UnweightedParticleEnsemble{T} <: UnweightedParticleRepresentation{Vector{T}}
    positions::Matrix{T}
end


UnweightedParticleEnsemble(vec::Vector{T}, N::Int) where T<:Number = UnweightedParticleEnsemble(repeat(vec, 1, N))









particle_dim(ens::UnweightedParticleEnsemble)    = size(ens.positions, 1)









no_of_particles(ens::UnweightedParticleEnsemble) = size(ens.positions, 2)









get_pos(ens::UnweightedParticleEnsemble, i) = view(ens.positions, :, i)









Base.show(io::IO, ens::UnweightedParticleEnsemble) = print(io, "Unweighted particle ensemble
    # of particles: ", no_of_particles(ens),"
    particle type:  ", particle_dim(ens),"-dimensional ", eltype(ens))







function UnweightedParticleEnsemble(model::HiddenStateModel, N::Int)
    return UnweightedParticleEnsemble(hcat([initialize(model) for i in 1:N]...))
end







function propagate!(ens::UnweightedParticleEnsemble{S}, model::HiddenStateModel{Vector{S}, ContinuousTime}, dt) where S
    propagate!(ens.positions, model, dt)
end







mean(ens::UnweightedParticleEnsemble) = Statistics.mean(ens.positions, dims=2)
cov(ens::UnweightedParticleEnsemble)  = Statistics.cov(ens.positions, dims=2, corrected=false)
var(ens::UnweightedParticleEnsemble)  = Statistics.var(ens.positions, dims=2, corrected=false)