
"""
    DifferentialRKHSMethodS1(epsilon, lambda)

Differential loss reproducing kernel Hilbert space (RKHS) method on the circle, adapted from [1], Section III.
The kernel is a von Mises kernel.

### Parameters

- `kappa`:  scale parameter of the von Mises kernel. Large values produce a narrow kernel.
- `lambda`: regularization parameter, Eq. 20 in [1]

### Reference

[1] Radhakrishnan, A. & and Meyn, S. (2018). Feedback particle filter design using a differential-loss reproducing kernel Hilbert space. 2018 Annual American Control Conference (ACC). IEEE. https://doi.org/10.23919/ACC.2018.8431689
"""
struct DifferentialRKHSMethodS1 <: GainEstimationMethod
    kappa::Float64
    lambda::Float64
end


# Helper functions for DifferentialRKHSMethodS1
function DifferentialRKHS_MakeMatricesS1(vec::AbstractVector, kappa::Number)
    # constructs kernel matrices for DifferentialRKHSMethodS1
    L = length(vec)

    mat = zeros(Float64, L, L, 3)

    @inbounds @simd for i in 1:L
                mat[i,i,1] = exp(kappa)
                mat[i,i,3] = kappa * mat[i,i,1]
                @simd for j in i+1:L
                    diff = vec[i] - vec[j]
                    cosd = cos(diff)
                    sind = sin(diff)
                    K    = exp(kappa * cosd)
                    Kx   = -kappa * sind * K
                    Kxy  =  kappa * (cosd - kappa * sind^2) * K
                    mat[i,j,1] = mat[j,i,1] = K
                    mat[i,j,2] = Kx
                    mat[j,i,2] = -Kx
                    mat[i,j,3] = mat[j,i,3] = Kxy
                end
    end

    return view(mat, :, :, 1), view(mat, :, :, 2), view(mat, :, :, 3)
end


import FeedbackParticleFilters.solve!
function solve!(eq::PoissonEquation, method::DifferentialRKHSMethodS1)
    kappa = method.kappa
    N     = no_of_particles(eq)
    lamb  = N * method.lambda

    H = Htilde(eq)

    # Gaussian kernel and partial derivative matrices
    K, Kx, Kxy = DifferentialRKHS_MakeMatricesS1(eq.positions[1,:], kappa)

    # compute M and b
    M, b = DifferentialRKHS_M_and_b!(K, Kx, Kxy, H', lamb) # warning: this function multiplies K, Kx, and Kxy by lamb

    # solve linear system
    beta = M \ b

    # write potential and gain
    eq.potential  .= ( K*view(beta, 1:N, :) - Kx*view(beta, N+1:2*N, :) )' / lamb
    broadcast!(-, eq.potential, eq.potential, Statistics.mean(eq.potential, dims=2))
    eq.gain[1,:,:] = ( -Kx * view(beta, 1:N, :) + Kxy * view(beta, N+1:2*N, :) ) / lamb
end