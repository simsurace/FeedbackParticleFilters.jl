@doc raw"""
    LinearDiffusionStateModel(f::Function, g::Function; init)

A diffusion process hidden state model ``dX_t = f(X_t)dt + g(X_t)dW_t``, where f is the `drift_function`, g is the `observation_function`, X_t is the `n`-dimensional hidden state at time t, and W_t is an `m`-dimensional Brownian motion process.

Optional argument `init` stands for the initial condition of the process, which is either
* A vector of length `n` for a fixed (deterministic) initial condition
* A `Distributions.Sampleable` type for a random initial condition

If argument `init` is left out, it is set to either 
* a multivariate normal distribution with covariance matrix set to the stationary variance, if it exists
* the zero vector
"""
struct LinearDiffusionStateModel{S, F1<:AbstractMatrix{S}, F2<:AbstractMatrix{S}, TI} <: ContinuousTimeHiddenStateModel{Vector{S}, ContinuousTime}
    drift_matrix::F1
    diffusion_matrix::F2
    init::TI
    function LinearDiffusionStateModel(A::AbstractMatrix{T}, B::AbstractMatrix{T}, init::TI) where TI<:Union{Distributions.Sampleable, AbstractVector{T}} where T
        n       = size(A, 1)
        m       = size(B, 2)
        size(B, 1) == n || throw(DimensionMismatch("Diffusion matrix has incorrect number of rows."))
        if init isa Distributions.Sampleable && eltype(rand(init)) == eltype(A)
            return new{eltype(A), typeof(A), typeof(B), typeof(init)}(A, B, init)
        else
            error("Error: initial condition and drift and diffusion matrices have incompatible element types.")
        end     
    end
end


function LinearDiffusionStateModel(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where T
    n       = size(A, 1)
    m       = size(B, 2)
    size(B, 1) == n || throw(DimensionMismatch("Diffusion matrix has incorrect number of rows."))
        
    mu      = zeros(T, n)
    Sigma   = find_stationary_variance(A, BB)
    if Sigma == zeros(T, n, n)
        return LinearDiffusionStateModel(A, B, mu)
    else
        return LinearDiffusionStateModel(A, B, Distributions.MvNormal(mu, Sigma))
    end
end
        
        
function find_stationary_variance(A::AbstractMatrix{T}, BB::AbstractMatrix{T})
    Sigma   = zeros(T, n, n)
    epsilon = 1.
    temp    = zeros(T, n, n)
    count   = 0
    while epsilon > 1e-4 && count < 10000 
        temp   .= Sigma
        Sigma  += (BB + A * Sigma + Sigma * A')/100
        epsilon = LinearAlgebra.norm(Sigma - temp)
        count += 1
    end
    if count == 10000 || Sigma != Sigma
        @warn "Iterative procedure to find stationary variance did not converge within 10000 iterations. Most likely the process does not have a stationary variance. Proceeding with initial variance set to zero."
        return zeros(T, n, n)
    else
        return Sigma/2 + Sigma'/2
    end
end
    
    
    

#####################
### BASIC METHODS ###
#####################
    
    
    

        
        

drift(model::LinearDiffusionStateModel)             = model.drift_matrix
diffusion(model::LinearDiffusionStateModel)         = model.diffusion_matrix    
initial_condition(model::LinearDiffusionStateModel) = model.init
dim(model::LinearDiffusionStateModel)               = size(drift(model), 1)
dimW(model::LinearDiffusionStateModel)              = size(diffusion(model), 2)
    
    
    
    
    
    

    
function Base.show(io::IO, model::LinearDiffusionStateModel)
    print(io, "Linear diffusion process model for the hidden state
    type of hidden state:                   ", dim(model),"-dimensional vector
    number of independent Brownian motions: ", dimW(model),"
    initial condition:                      ", model.init isa Distributions.Sampleable ? "random" : "fixed")
end
    
    
    
    
######################
### TIME EVOLUTION ###
######################
    
    
    
    
    
    
    
(model::LinearDiffusionStateModel)(x::AbstractVector, dt) = x + dt*model.drift_matrix*x + sqrt(dt)*model.diffusion_matrix*randn(eltype(x), dimW(model))
(model::LinearDiffusionStateModel)(x::AbstractMatrix, dt) = x + dt*model.drift_matrix*x + sqrt(dt)*model.diffusion_matrix*randn(eltype(x), dimW(model), size(x, 2))
    
    
    
    
    
    
    
    
function propagate!(x::AbstractVector, model::LinearDiffusionStateModel, dt)
    x .+= dt .* model.drift_matrix * x + sqrt(dt) .* model.diffusion_matrix * randn(eltype(x), dimW(model))
    return x
end 

function propagate!(x::AbstractMatrix, model::LinearDiffusionStateModel, dt)
    x .+= dt .* model.drift_matrix * x + sqrt(dt) .* model.diffusion_matrix * randn(eltype(x), dimW(model), size(x, 2))
    return x
end 