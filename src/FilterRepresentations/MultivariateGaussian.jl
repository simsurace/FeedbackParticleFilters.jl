@doc raw"""
    MultivariateGaussian(mean, cov)

A representation of a multivariate normal distribution with specified mean
and covariance matrix.
"""
struct MultivariateGaussian{S, TM, TP} <: ParametricRepresentation{Vector{S}, S}
    mean::TM
    cov::TP
    MultivariateGaussian(mean::AbstractVector{S}, cov::AbstractMatrix{S}) where S = new{S, typeof(mean), typeof(cov)}(mean, cov)
end

mean(gauss::MultivariateGaussian) = gauss.mean
cov(gauss::MultivariateGaussian)  = gauss.cov
var(gauss::MultivariateGaussian)  = LinearAlgebra.diag(gauss.cov)

parameter_vector(gauss::MultivariateGaussian) = vcat(mean(gauss), vec(cov(gauss)))
