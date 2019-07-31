@doc raw"""
    UnweightedParticleEnsemble{T}

An ensemble of `N` particles, each of dimension `n`.
"""
struct MultivariateGaussian{TM<:AbstractVector{S}, TP<:AbstractVector{S}} <: ParametricRepresentation{Vector{S}, S}
    mean::TM
    cov::TP
end

mean(gauss::MultivariateGaussian) = gauss.mean
cov(gauss::MultivariateGaussian)  = gauss.cov

parameter_vector(gauss::MultivariateGaussian) = vcat(mean(gauss), vec(cov(gauss)))