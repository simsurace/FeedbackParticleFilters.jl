@doc raw"""
    DiffusionStateModel(f::Function, g::Function, init)

A diffusion process hidden state model ``dX_t = f(X_t)dt + g(X_t)dW_t``, where f is the `drift_function`, g is the `observation_function`, X_t is the `n`-dimensional hidden state at time t, and W_t is an `m`-dimensional Brownian motion process.

Argument `init` stands for the initial condition of the process, which is either
* A vector of length `n` for a fixed (deterministic) initial condition
* A `Distributions.Sampleable` type for a random initial condition
"""
struct DiffusionStateModel{S, F1, F2, TI} <: HiddenStateModel{Vector{S}, ContinuousTime}
    n::Int
    m::Int
    drift_function::F1
    diffusion_function::F2
    init::TI
    function DiffusionStateModel(f::Function, g::Function, init::TI) where TI<:Union{Distributions.Sampleable, Any}
        if init isa Distributions.Sampleable 
            x = rand(init)
        else
            x = init
        end
            
        F = f(x)
        G = g(x)
            
        if length(f(x)) == size(g(x), 1)
            n = length(f(x)) 
        else 
            error("Error: drift_function and diffusion_function have incompatible output sizes.")
        end
                
        m = size(g(x), 2)
                
        if eltype(x) == eltype(F) == eltype(G)
            S = eltype(x)
            if x isa Vector{T}
                return new{S, typeof(f), typeof(g), typeof(init)}(n, m, f, g, init)
            else
                error("Error: initial condition has incorrect type.")
            end
        else
            error("Error: initial condition and outputs of drift_function and diffusion_function have incompatible element types.")
        end
    end
end
               

                
                

#####################
### BASIC METHODS ###
#####################
                
                
                
                
                
                
                
state_dim(model::DiffusionStateModel) = model.n
noise_dim(model::DiffusionStateModel) = model.m

                
                
                
                
                
                
                           
initial_condition(model::DiffusionStateModel{T, F1, F2, TI}) where {F1, F2, TI<:Distributions.Sampleable}      = rand(model.init)

                
                
                
                
                
                
                
                
function Base.show(io::IO, model::DiffusionStateModel)
    print(io, "Diffusion process model for the hidden state
    type of hidden state:                   ", state_dim(model),"-dimensional vector
    number of independent Brownian motions: ", noise_dim(model),"
    initial condition:                      ", model.init isa Distributions.Sampleable ? "random" : "fixed")
end
                
                
                
                
######################
### TIME EVOLUTION ###
######################


                
                
                
                
                
(model::DiffusionStateModel)(x::AbstractVector, dt) = x + model.drift_function(x)*dt + sqrt(dt)*model.diffusion_function(x)*randn(eltype(x), dimW(model))


                
                
                
                
                
                
                
function (model::DiffusionStateModel)(x::AbstractMatrix, dt)
    N   = size(x, 2)
    F   = model.drift_function(x)
    G   = model.diffusion_function(x)
    out = zeros(eltype(x), state_dim(model), N)
    sqr = sqrt(dt)
    @inbounds for b in 1:N
        for c in 1:noise_dim(model), a in 1:state_dim(model)
            out[a,b]  +=  x[a,b]  +  dt * F[a,b]  +  sqr * G[a,c,b] * randn(eltype(x))
        end
    end
    return out
end 
                
                
                
                
                
                
                
function propagate!(x::AbstractVector, model::DiffusionStateModel, dt)
    N   = size(x, 2)
    F   = model.drift_function(x)
    G   = model.diffusion_function(x)
    sqr = sqrt(dt)
    @inbounds for c in 1:noise_dim(model), a in 1:state_dim(model)
            x[a]  +=  dt * F[a]  +  sqr * G[a,c] * randn(eltype(x))
    end
    return x
end 

                
                
                
                
                
                
                
                
                
function propagate!(x::AbstractMatrix, model::DiffusionStateModel, dt)
    N   = size(x, 2)
    F   = model.drift_function(x)
    G   = model.diffusion_function(x)
    sqr = sqrt(dt)
    @inbounds for b in 1:N
        for c in 1:noise_dim(model), a in 1:state_dim(model)
            x[a,b]  +=  dt * F[a,b]  +  sqr * G[a,c,b] * randn(eltype(x))
        end
    end
    return x
end 