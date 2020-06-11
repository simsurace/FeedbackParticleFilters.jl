@doc raw"""
    WeightedParticleEnsemble{T}

An ensemble of `N` particles, each of dimension `n`.
"""
struct WeightedParticleEnsemble{T} <: WeightedParticleRepresentation{Vector{T}}
    positions::Matrix{T}
    weights::StatsBase.ProbabilityWeights
end

particle_dim(ens::WeightedParticleEnsemble)    = size(ens.positions, 1)









no_of_particles(ens::WeightedParticleEnsemble) = size(ens.positions, 2)









get_pos(ens::WeightedParticleEnsemble, i) = view(ens.positions, :, i)









get_weight(ens::WeightedParticleEnsemble, i) = view(ens.weights, i)









sum_of_weights(ens::WeightedParticleEnsemble) = sum(ens.weights)









Base.show(io::IO, ens::WeightedParticleEnsemble) = print(io, "Weighted particle ensemble
    # of particles: ", no_of_particles(ens),"
    particle type:  ", particle_dim(ens),"-dimensional ", eltype(ens))







function WeightedParticleEnsemble(model::HiddenStateModel, N::Int)
    return WeightedParticleEnsemble(hcat([initialize(model) for i in 1:N]...), StatsBase.ProbabilityWeights(fill(1/N, N)))
end







function propagate!(ens::WeightedParticleEnsemble{S}, model::HiddenStateModel{Vector{S}, ContinuousTime}, dt) where S
    propagate!(ens.positions, model, dt)
end







mean(ens::WeightedParticleEnsemble) = Statistics.mean(ens.positions, ens.weights, dims=2)
cov(ens::WeightedParticleEnsemble)  = Statistics.cov(ens.positions, ens.weights, 2, corrected=false)
var(ens::WeightedParticleEnsemble)  = Statistics.var(ens.positions, ens.weights, 2, corrected=false)







function resample!(ens::WeightedParticleEnsemble)
    N   = no_of_particles(ens)
    idx = StatsBase.sample(1:N, ens.weights, N)
    ens.positions .= view(ens.positions, :, idx)
    ens.weights   .= 1/N
    return ens.positions
end