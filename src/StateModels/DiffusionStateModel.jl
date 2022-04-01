@doc raw"""
    DiffusionStateModel(f::Function, g::Function, init)

Returns a diffusion process hidden state model ``dX_t = f(X_t)dt + g(X_t)dW_t``, where ``f`` is the `drift_function`, ``g`` is the `observation_function`, ``X_t`` is the ``n``-dimensional hidden state at time ``t``, and ``W_t`` is an ``m``-dimensional Brownian motion process.

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
            x isa Vector || (x = [x])
        else
            x = init
        end
            
        F = f(x)
        G = g(x)
            
        if length(F) == length(x)
            n = length(F)
        else 
            error("Error: output of drift_function and initial condition have incompatible lengths.")
        end
                
        if length(F) != size(G, 1)
            error("Error: drift_function and diffusion_function have incompatible output sizes.")
        end
                
        m = size(G, 2)
                
        if eltype(x) == eltype(F) == eltype(G)
            S = eltype(x)
            if x isa Vector
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
            
function drift(model::DiffusionStateModel)
    if hasmethod(model.drift_function, Tuple{AbstractMatrix})
        return model.drift_function
    else
        f(x) = model.drift_function(x)
        f(x::AbstractMatrix) = mapslices(f, x, dims=1)
        return f
    end
end
               
function diffusion(model::DiffusionStateModel)         
    if hasmethod(model.diffusion_function, Tuple{AbstractMatrix})
        return model.diffusion_function
    else
        g(x) = model.diffusion_function(x)                       
        g(x::AbstractMatrix) = cat([model.diffusion_function(col) for col in eachcol(x)]..., dims=3)
        return g
    end
end
                        
initial_condition(model::DiffusionStateModel) = model.init
                        
state_dim(model::DiffusionStateModel) = model.n
                   
                        
        
"""
    noise_dim(model)

Returns the dimension of the Brownian motion ``W_t`` in the diffusion model ``dX_t = f(X_t)dt + g(X_t)dW_t``.
"""                        
noise_dim(model::DiffusionStateModel) = model.m
function initialize(model::DiffusionStateModel{T, F1, F2, TI}) where {T, F1, F2, TI<:Distributions.Sampleable}
    x = rand(model.init)
    if x isa Vector
        return x
    else
        return [x]
    end
end
                            
function initialize(model::DiffusionStateModel{T, F1, F2, TI}) where 
    TI<:AbstractVector{T} where {T, F1, F2}
    return model.init
end
                
   
                            
                            
"""
    drift_function(model)

Returns the drift function ``f`` of the diffusion model ``dX_t = f(X_t)dt + g(X_t)dW_t``.
"""                
drift_function(model::DiffusionStateModel) = drift(model)
                            
                            
                            

"""
    diffusion_function(model)

Returns the diffusion function ``g`` of the diffusion model ``dX_t = f(X_t)dt + g(X_t)dW_t``.
"""                            
diffusion_function(model::DiffusionStateModel) = diffusion(model)
        


                
                
                            
                
function Base.show(io::IO, ::MIME"text/plain", model::DiffusionStateModel)
    print(io, "Diffusion process model for the hidden state
    type of hidden state:                   ", state_dim(model),"-dimensional vector
    number of independent Brownian motions: ", noise_dim(model),"
    initial condition:                      ", model.init isa Distributions.Sampleable ? "random" : "fixed")
end

function Base.show(io::IO, model::DiffusionStateModel)
    print(io, state_dim(model),"-dimensional diffusion with ", noise_dim(model), "-dimensional Brownian motion")
end

function Base.show(io::IO, ::MIME"text/plain", model::Type{DiffusionStateModel{S1, S2, S3, S4}}) where {S1, S2, S3, S4}
    print(io, "DiffusionStateModel{", S1, ",", S2, ",", S3, ",", S4, "}")
end

function Base.show(io::IO, model::Type{DiffusionStateModel{S1, S2, S3, S4}}) where {S1, S2, S3, S4}
    print(io, "DiffusionStateModel")
end
                
                
                
                
                
                
                
                
                
                
                
                
######################
### TIME EVOLUTION ###
######################


                
                
                
                
                
(model::DiffusionStateModel)(x::AbstractVector, dt) = x + model.drift_function(x)*dt + sqrt(dt)*model.diffusion_function(x)*randn(eltype(x), noise_dim(model))


                
                
                
                
                
                
                
function (model::DiffusionStateModel)(x::AbstractMatrix, dt)
    N   = size(x, 2)
    F   = drift_function(model)(x)
    G   = diffusion_function(model)(x)
    if ndims(G) == 2
        G = repeat(G, 1, 1, N)
    end
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
    F   = drift_function(model)(x)
    G   = diffusion_function(model)(x)
    sqr = sqrt(dt)
    @inbounds for c in 1:noise_dim(model), a in 1:state_dim(model)
            x[a]  +=  dt * F[a]  +  sqr * G[a,c] * randn(eltype(x))
    end
    return x
end 

                
                
                
                
                
                
                
                
                
function propagate!(x::AbstractMatrix, model::DiffusionStateModel, dt)
    N   = size(x, 2)
    F   = drift_function(model)(x)
    G   = diffusion_function(model)(x)
    if ndims(G) == 2
        G = repeat(G, 1, 1, N)
    end
    sqr = sqrt(dt)
    @inbounds for b in 1:N
        for c in 1:noise_dim(model)
            for a in 1:state_dim(model)
                x[a,b]  +=  dt * F[a,b]  +  sqr * G[a,c,b] * randn(eltype(x))
            end
        end
    end
    return x
end 
                        
                        
                        
                        
                        
                        
################################
### CONVENIENCE CONSTRUCTORS ###
################################
                            
@doc raw"""
    ScalarDiffusionStateModel(f::Function, g::Function, init)

Returns a scalar diffusion process hidden state model ``dX_t = f(X_t)dt + g(X_t)dW_t``.
"""                            
function ScalarDiffusionStateModel(f::Function, g::Function, init::ContinuousUnivariateDistribution)
    F(x::AbstractVector) = [f(x[1])]
    F(x::AbstractMatrix) = mapslices(F, x, dims=1)
    G(x::AbstractVector) = g(x[1])*ones(1,1)
    G(x::AbstractMatrix) = permutedims(repeat(mapslices(y->g(y[1]), x, dims=1), 1, 1, 1), [1,3,2])
    return DiffusionStateModel(F, G, init)
end
