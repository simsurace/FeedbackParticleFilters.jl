#####################
# Semigroup methods #
#####################







"""
    SemigroupMethod

Semigroup method from Algorithm 1 in [1].

Concrete methods: SemigroupMethod1d, RegularizedSemigroupMethod1d

[1] Taghvaei, A., & Mehta, P. G. (2016). Gain function approximation in the feedback particle filter. In 2016 IEEE 55th Conference on Decision and Control (CDC) (pp. 5446–5452). IEEE. https://doi.org/10.1109/CDC.2016.7799105
"""
abstract type SemigroupMethod <: GainEstimationMethod end










"""
    SemigroupMethod1d(epsilon, delta, max_iter(=100))

Semigroup method from Algorithm 1 in [1].

### Parameters

- `epsilon`: small time parameter for the diffusion kernel
- `delta`: tolerance for the fixed point equation
- `max_iter(=100)`: maximum number of iterations for the fixed point equation

### References

[1] Taghvaei, A., & Mehta, P. G. (2016). Gain function approximation in the feedback particle filter. In 2016 IEEE 55th Conference on Decision and Control (CDC) (pp. 5446–5452). IEEE. https://doi.org/10.1109/CDC.2016.7799105
"""
struct SemigroupMethod1d <: SemigroupMethod
    epsilon::Float64
    delta::Float64
    max_iter::Int
end
SemigroupMethod1d(epsilon::Float64, delta::Float64) = SemigroupMethod1d(epsilon, delta, 100)

function Solve!(eq::ScalarPoissonEquation, method::SemigroupMethod1d)
    N = length(eq.positions)
    H = copy(eq.H)
    broadcast!(-, H, H, eq.mean_H)
    broadcast!(*, H, H, method.epsilon)

    # compute T operator
    T = zeros(Float64, N, N)
    for i in 1:N
        T[i,i] = 1.0
        for j in i+1:N
            T[i,j] = exp(-(eq.positions[i]-eq.positions[j])^2/(4*method.epsilon))
            T[j,i] = T[i,j]
        end
    end
    broadcast!(/, T, T, sqrt.(sum(T,dims=1) .* sum(T,dims=2)))
    broadcast!(/, T, T, sum(T,dims=2))

    # solve fixed-point equation for potential
    newpotential = copy(eq.potential)::Array{Float64,1}
    fluctuation = 1.
    n = 1
    while fluctuation > method.delta
        LinearAlgebra.mul!(newpotential, T, eq.potential)
        broadcast!(+, newpotential, newpotential, H)
        broadcast!(-, newpotential, newpotential, StatsBase.mean(newpotential))
        fluctuation = maximum(abs.(newpotential - eq.potential))
        eq.potential .= newpotential
        n += 1
        if n == method.max_iter
            print("!")
            break
        end
    end

    # compute gain from potential
    eq.gain .= T * (eq.potential .* eq.positions) - (T*eq.potential) .* (T*eq.positions)
    broadcast!(/, eq.gain, eq.gain, 2*method.epsilon)
end









"""
    RegularizedSemigroupMethod1d(epsilon, delta, max_iter(=100), lambda(=1E-3))

Like `SemigroupMethod1d`, but in addition noise of amplitude `lambda` is added to the diagonal of the semigroup operator T in order to improve convergence of the fixed point equation.
"""
struct RegularizedSemigroupMethod1d <: SemigroupMethod
    epsilon::Float64
    delta::Float64
    max_iter::Int
    lambda::Float64
end
RegularizedSemigroupMethod1d(epsilon::Float64, delta::Float64) = SemigroupMethod1d(epsilon, delta, 100, 1E-3)

function Solve!(eq::ScalarPoissonEquation, method::RegularizedSemigroupMethod1d)
    N = length(eq.positions)
    H = copy(eq.H)
    broadcast!(-, H, H, eq.mean_H)
    broadcast!(*, H, H, method.epsilon)

    # compute T operator
    T = zeros(Float64, N, N)
    for i in 1:N
        T[i,i] = 1.0
        for j in i+1:N
            T[i,j] = exp(-(eq.positions[i]-eq.positions[j])^2/(4*method.epsilon))
            T[j,i] = T[i,j]
        end
    end
    broadcast!(/, T, T, sqrt.(sum(T,dims=1) .* sum(T,dims=2)))
    broadcast!(/, T, T, sum(T,dims=2))

    # add noise to regularize T
    for i in 1:N
        T[i,i] -=  method.lambda * rand(Distributions.Uniform(0.9,1))
    end

    # solve fixed-point equation for potential
    newpotential = copy(eq.potential)::Array{Float64,1}
    fluctuation = 1.
    n = 1
    while fluctuation > method.delta
        LinearAlgebra.mul!(newpotential, T, eq.potential)
        broadcast!(+, newpotential, newpotential, H)
        broadcast!(-, newpotential, newpotential, StatsBase.mean(newpotential))
        fluctuation = maximum(abs.(newpotential - eq.potential))
        eq.potential .= newpotential
        n += 1
        if n == method.max_iter
            print("!")
            break
        end
    end

    # compute gain from potential
    eq.gain .= T * (eq.potential .* eq.positions) - (T*eq.potential) .* (T*eq.positions)
    broadcast!(/, eq.gain, eq.gain, 2*method.epsilon)
end



#########################
# Empirical Risk Method #
#########################







"""
    EmpiricalRiskMinimizationMethod

Any gain estimation method that tries to minimize an empirical loss function between true and estimated gain.
"""
abstract type EmpiricalRiskMinimizationMethod <: GainEstimationMethod end




"""
    DifferentialRKHSMethod1d(epsilon, lambda)

Differential loss reproducing kernel Hilbert space (RKHS) method from [1], Section III.

### Parameters

- `epsilon`: variance of the Gaussian kernel, Eq. 17 in [1]
- `lambda`: regularization parameter, Eq. 20 in [1]

### Reference

[1] Radhakrishnan, A. & and Meyn, S. (2018). Feedback particle filter design using a differential-loss reproducing kernel Hilbert space. 2018 Annual American Control Conference (ACC). IEEE. https://doi.org/10.23919/ACC.2018.8431689
"""
struct DifferentialRKHSMethod1d <: EmpiricalRiskMinimizationMethod
    epsilon::Float64
    lambda::Float64
end


# Helper functions for DifferentialRKHSMethod1d
function DifferentialRKHS1d_MakeMatrices(vec::AbstractVector, eps::Number)
    # constructs kernel matrices for DifferentialRKHSMethod1d
    L = length(vec)
    eps1 = 2*eps
    eps2 = eps^2

    mat = zeros(Float64, L, L, 3)

    @inbounds @simd for i in 1:L
                mat[i,i,1] = 1
                mat[i,i,3] = 1/eps
                @simd for j in i+1:L
                    diff = vec[j] - vec[i]
                    sq = diff*diff
                    K = exp(-sq/eps1)
                    Kx = diff*K/eps
                    Kxy = (eps - sq) * K / eps2
                    mat[i,j,1] = mat[j,i,1] = K
                    mat[i,j,2] = Kx
                    mat[j,i,2] = -Kx
                    mat[i,j,3] = mat[j,i,3] = Kxy
                end
    end

    return view(mat, :, :, 1), view(mat, :, :, 2), view(mat, :, :, 3)
end



function DifferentialRKHS1d_M_and_b!(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix, v::AbstractVector, lamb::Number)
    # constructs matrix M and vector b for DifferentialRKHSMethod1d
    L = size(A,1)

    b = zeros(Float64, 2*L)
    LinearAlgebra.mul!(view(b,   1:L  ), A, v)
    LinearAlgebra.mul!(view(b, L+1:2*L), B, v)


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


function Solve!(eq::ScalarPoissonEquation, method::DifferentialRKHSMethod1d)
    eps = method.epsilon
    N = length(eq.positions)
    lamb = N * method.lambda

    H = copy(eq.H)
    broadcast!(-, H, H, eq.mean_H)

    # Gaussian kernel and partial derivative matrices
    K, Kx, Kxy = DifferentialRKHS1d_MakeMatrices(eq.positions, eps)

    # compute M and b
    M, b = DifferentialRKHS1d_M_and_b!(K, Kx, Kxy, H, lamb) # warning: this function multiplies K, Kx, and Kxy by lamb

    # solve linear system
    beta = M \ b

    # write potential and gain
    eq.potential = ( K*view(beta, 1:N) - Kx*view(beta, N+1:2*N) ) / lamb
    broadcast!(-, eq.potential, eq.potential, StatsBase.mean(eq.potential))
    eq.gain = ( Kx * view(beta, 1:N) + Kxy * view(beta, N+1:2*N) ) / lamb
end
