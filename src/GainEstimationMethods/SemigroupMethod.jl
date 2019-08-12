"""
    SemigroupMethod(epsilon, delta, max_iter)

Semigroup method from Algorithm 1 in [1].

[1] Taghvaei, A., & Mehta, P. G. (2016). Gain function approximation in the feedback particle filter. In 2016 IEEE 55th Conference on Decision and Control (CDC) (pp. 5446–5452). IEEE. https://doi.org/10.1109/CDC.2016.7799105

    SemigroupMethod(epsilon, delta, max_iter, lambda)

Semigroup method with regularization parameter `lambda`.
"""
struct SemigroupMethod <: GainEstimationMethod
    epsilon::Float64
    delta::Float64
    max_iter::Int
    lambda::Float64
    SemigroupMethod(eps, delta, max_iter, lambda=0.) = new(eps, delta, max_iter, lambda)
end

SemigroupMethod(epsilon::Float64, delta::Float64) = SemigroupMethod(epsilon, delta, 100)






function solve!(eq::PoissonEquation, method::SemigroupMethod)
    N = length(eq.positions)
    H = Htilde(eq)
    broadcast!(*, H, H, method.epsilon)

    # compute T operator
    T = zeros(eltype(eq.positions), N, N)
    n = state_dim(eq)
    for i in 1:N
        T[i,i] = one(eltype(eq.positions))
        for j in i+1:N
            sq = zero(eltype(eq.positions))
            for l in 1:n
                sq += (eq.positions[l,i] - eq.positions[l,j])^2
            end
            T[i,j] = exp(-sq/(4*method.epsilon))
            T[j,i] = T[i,j]
        end
    end
    broadcast!(/, T, T, sqrt.(sum(T,dims=1) .* sum(T,dims=2)))
    broadcast!(/, T, T, sum(T,dims=2))
    
    # add noise to regularize T
    if method.lambda > 0.
        for i in 1:N
            T[i,i] -=  method.lambda * rand(Distributions.Uniform(0.9,1))
        end
    end

    # solve fixed-point equation for potential
    newpotential = copy(eq.potential)::Array{Float64,2}
    fluctuation = 1.
    n = 1
    while fluctuation > method.delta
        LinearAlgebra.mul!(newpotential, eq.potential, T')
        broadcast!(+, newpotential, newpotential, H)
        broadcast!(-, newpotential, newpotential, Statistics.mean(newpotential, dims=2))
        fluctuation = maximum(abs.(newpotential - eq.potential))
        eq.potential .= newpotential
        n += 1
        if n == method.max_iter
            print("!")
            break
        end
    end

    # compute gain from potential
    gainhelper_semigroup!(eq, T) 
    broadcast!(/, eq.gain, eq.gain, 2*method.epsilon)
end

function gainhelper_semigroup!(eq::PoissonEquation, T::AbstractMatrix)
    pos     = eq.positions
    pot     = eq.potential
    gain    = eq.gain
    
    n       = state_dim(eq)
    m       = obs_dim(eq)
    N       = no_of_particles(eq)
    
    @inbounds for l in 1:n, k in 1:m, i in 1:N
        Tpotpos = zero(eltype(gain))
        Tpot    = zero(eltype(gain))
        Tpos    = zero(eltype(gain))
        for j in 1:N
            Tpot        += T[i,j] * pot[k,j]
            Tpos        += T[i,j] * pos[l,j]
            Tpotpos     += T[i,j] * pot[k,j] * pos[l,j]
        end
        gain[l,i,k] = Tpotpos - Tpot * Tpos
    end
end