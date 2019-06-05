########################
### Semigroup method ###
########################

"""
    SemigroupMethod

Semigroup method from Algorithm 1 in [1].

Concrete methods: SemigroupMethod1d, RegularizedSemigroupMethod1d

[1] Taghvaei, A., & Mehta, P. G. (2016). Gain function approximation in the feedback particle filter. In 2016 IEEE 55th Conference on Decision and Control (CDC) (pp. 5446–5452). IEEE. https://doi.org/10.1109/CDC.2016.7799105
"""
abstract type SemigroupMethod <: GainEstimationMethod end

"""
    SemigroupMethod1d(epsilon, delta, max_iter(=100))

One-dimensional semigroup method with Gaussian kernels of variance `epsilon`. 
The fixed-point equation is iterated for a maximum of `max_iter` iterations as long as the maximum change in the potential is larger than `delta`.
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
    SemigroupMethod1d(epsilon, delta, max_iter(=100), lambda(=1E-3))

One-dimensional semigroup method with Gaussian kernels of variance `epsilon`. 
The fixed-point equation is iterated for a maximum of `max_iter` iterations as long as the maximum change in the potential is larger than `delta`.
Noise of amplitude `lambda` is added to the diagonal of the semigroup operator T in order to improve convergence of the fixed point equation.
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