struct FPF{T} <: AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}
    gain_method::T
end







struct FPFState{T, TF}
    ensemble::UnweightedParticleEnsemble{T}
    eq::PoissonEquation{T, TF}
    function FPFState(ens::UnweightedParticleEnsemble{T}, eq::PoissonEquation{T, TF}) where {T, TF}
        update!(eq, ens)
        new{T, TF}(ens, eq)
    end
end












# helper functions

function gainxerror(gain::Array{T, 3}, error::Array{T, 2}) where T
    size(gain, 2) == size(error, 2) ? N = size(gain, 2) : throw(DimensionMismatch("The provided gain and error are for different numbers of particles."))
    size(gain, 3) == size(error, 1) ? m = size(gain, 3) : throw(DimensionMismatch("The provided gain and error are for different numbers of observed variables."))
    out = zeros(T, size(gain, 1), size(gain, 2))
    @inbounds for k in 1:N, i in 1:size(gain, 1), j in 1:m
            out[i, k] += gain[i, k, j] * error[j, k]
    end
    return out
end

function add_gainxerror!(out::Array{T, 2}, gain::Array{T, 3}, error::Array{T, 2}) where T
    size(gain, 2) == size(error, 2) == size(out, 2) ? N = size(gain, 2) : throw(DimensionMismatch("The provided gain, error, and output array are for different numbers of particles."))
    size(gain, 3) == size(error, 1)                 ? m = size(gain, 3) : throw(DimensionMismatch("The provided gain and error are for different numbers of observed variables."))
    size(gain, 1) == size(out, 1)                   ? n = size(gain, 1) : throw(DimensionMismatch("The provided gain and output array are for different numbers of hidden variables."))
    @inbounds for k in 1:N, i in 1:n, j in 1:m
            out[i, k] += gain[i, k, j] * error[j, k]
    end
    return out
end

function applygain!(ens::UnweightedParticleEnsemble, eq::GainEquation, error::AbstractMatrix)
    add_gainxerror!(ens.positions, gain, error)
end





# time evolution

function propagate!(filter_state::FPFState, state_model, filt_algo::FPF, dt)
    propagate!(filter_state.ensemble, state_model, dt)
end

function assimilate!(dY, filter_state::FPFState, filt_algo::FPF, dt)
    ensemble = filter_state.ensemble
    eq       = filter_state.eq
    method   = filt_algo.gain_method
    
    error    = dY .- eq.H/2 .- eq.mean_H/2
    solve!(eq, method)
    applygain!(ensemble, eq, error)
    update!(eq, ensemble)
    return ensemble.positions
end

