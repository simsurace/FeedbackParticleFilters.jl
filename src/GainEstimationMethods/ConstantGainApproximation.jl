"""
    ConstantGainApproximation()

Represents an approximation of the gain by a constant (in the Euclidean sense) vector field, given by the covariance of the observation function and x under the particle distribution.
"""
struct ConstantGainApproximation <: GainEstimationMethod end

function solve!(eq::PoissonEquation, method::ConstantGainApproximation)
    n, N, m = size(eq.gain)
    htilde = Htilde(eq)
    @inbounds for j in 1:m
        for i in 1:n
            g = 0.
            for l in 1:N
                g += eq.positions[i, l] * htilde[j, l]
            end
            for k in 1:N
                eq.gain[i, k, j] = g/N
            end
        end
    end
    return eq.gain
end