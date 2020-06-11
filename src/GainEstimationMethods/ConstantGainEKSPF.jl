"""
    ConstantGainEKSPF()

Constant gain from [1], Algorithm 1.

### Reference

[1] Venugopal, M., Vasu, R. M., & Roy, D. (2016). An Ensemble Kushner-Stratonovich-Poisson Filter for Recursive Estimation in Nonlinear Dynamical Systems. IEEE Transactions on Automatic Control, 61(3), 823–828. https://doi.org/10.1109/TAC.2015.2450113
"""
struct ConstantGainEKSPF <: GainEstimationMethod end

function solve!(eq::PoissonEquation, method::ConstantGainEKSPF)
    n, N, m = size(eq.gain)
    htilde = Htilde(eq)
    @inbounds for j in 1:m
        for i in 1:n
            g = 0.
            for l in 1:N
                g += eq.positions[i, l] * htilde[j, l]
            end
            for k in 1:N
                eq.gain[i, k, j] = g / (N * eq.mean_H[j, 1])
            end
        end
    end
    return eq.gain
end