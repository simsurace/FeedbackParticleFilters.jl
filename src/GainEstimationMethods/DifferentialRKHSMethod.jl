"""
    DifferentialRKHSMethod(epsilon, lambda)

Differential loss reproducing kernel Hilbert space (RKHS) method from [1], Section III.

### Parameters

- `epsilon`: variance of the Gaussian kernel, Eq. 17 in [1]
- `lambda`: regularization parameter, Eq. 20 in [1]

### Reference

[1] Radhakrishnan, A. & and Meyn, S. (2018). Feedback particle filter design using a differential-loss reproducing kernel Hilbert space. 2018 Annual American Control Conference (ACC). IEEE. https://doi.org/10.23919/ACC.2018.8431689
"""
struct DifferentialRKHSMethod <: GainEstimationMethod
    epsilon::Float64
    lambda::Float64
end


# Helper functions for DifferentialRKHSMethod
function DifferentialRKHS_MakeMatrices(vec::AbstractVector, eps::Number)
    # constructs kernel matrices for DifferentialRKHSMethod
    L = length(vec)
    eps1 = 2*eps
    eps2 = eps^2

    mat = zeros(Float64, L, L, 3)

    @inbounds @simd for i in 1:L
                mat[i,i,1] = 1
                mat[i,i,3] = 1/eps
                @simd for j in i+1:L
                    diff = vec[j] - vec[i]
                    sq   = diff*diff
                    K    = exp(-sq/eps1)
                    Kx   = diff*K/eps
                    Kxy  = (eps - sq) * K / eps2
                    mat[i,j,1] = mat[j,i,1] = K
                    mat[i,j,2] = Kx
                    mat[j,i,2] = -Kx
                    mat[i,j,3] = mat[j,i,3] = Kxy
                end
    end

    return view(mat, :, :, 1), view(mat, :, :, 2), view(mat, :, :, 3)
end



function DifferentialRKHS_M_and_b!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, v::AbstractMatrix, lamb::Number)
    # constructs matrix M and vector b for DifferentialRKHSMethod
    L = size(A,1)
    m = size(v,2)

    b = zeros(Float64, 2*L, m)
    LinearAlgebra.mul!(view(b,   1:L  , :), A, v)
    LinearAlgebra.mul!(view(b, L+1:2*L, :), B, v)


    B2 = B*B
    BC = B*C
    C2 = C*C
    LinearAlgebra.lmul!(lamb, A)
    LinearAlgebra.lmul!(lamb, B)
    LinearAlgebra.lmul!(lamb, C)

    M = zeros(Float64, 2*L, 2*L)
    @simd for j in 1:L
        @simd for i in 1:L
            @inbounds M[i  ,j  ] =  A[i,j] - B2[i,j]
            @inbounds M[i+L,j  ] =  B[i,j] + BC[j,i]
            @inbounds M[i  ,j+L] = -B[i,j] - BC[i,j]
            @inbounds M[i+L,j+L] =  C[i,j] + C2[i,j]
        end
    end

    return M, b
end


function solve!(eq::PoissonEquation, method::DifferentialRKHSMethod)
    eps  = method.epsilon
    N    = no_of_particles(eq)
    lamb = N * method.lambda

    H = Htilde(eq)

    # Gaussian kernel and partial derivative matrices
    K, Kx, Kxy = DifferentialRKHS_MakeMatrices(eq.positions[1,:], eps)

    # compute M and b
    M, b = DifferentialRKHS_M_and_b!(K, Kx, Kxy, H', lamb) # warning: this function multiplies K, Kx, and Kxy by lamb

    # solve linear system
    beta = M \ b

    # write potential and gain
    eq.potential  .= ( K*view(beta, 1:N) - Kx*view(beta, N+1:2*N) )' / lamb
    broadcast!(-, eq.potential, eq.potential, Statistics.mean(eq.potential, dims=2))
    eq.gain[1,:,:] = ( Kx * view(beta, 1:N, :) + Kxy * view(beta, N+1:2*N, :) ) / lamb
end