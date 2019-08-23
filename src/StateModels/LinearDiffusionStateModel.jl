@doc raw"""
    LinearDiffusionStateModel(A, B; init)

Returns a linear diffusion process hidden state model ``dX_t = A X_t dt + B dW_t`` with appropriately sized matrices ``A`` and ``B``.

Optional argument `init` stands for the initial condition of the process, which is either
* A vector of length `n` for a fixed (deterministic) initial condition
* A `Distributions.Sampleable` type for a random initial condition

If argument `init` is left out, it is set to either 
* a multivariate normal distribution with covariance matrix set to the stationary variance, if it exists
* the zero vector
"""
struct LinearDiffusionStateModel{S, F1<:AbstractMatrix{S}, F2<:AbstractMatrix{S}, TI} <: HiddenStateModel{Vector{S}, ContinuousTime}
    drift_matrix::F1
    diffusion_matrix::F2
    init::TI
    function LinearDiffusionStateModel(A::AbstractMatrix{T}, B::AbstractMatrix{T}, init::TI) where TI<:Union{Distributions.Sampleable, AbstractVector{T}} where T
        n       = size(A, 1)
        m       = size(B, 2)
        size(B, 1) == n || throw(DimensionMismatch("Diffusion matrix has incorrect number of rows."))
        if init isa Distributions.Sampleable && eltype(rand(init)) == eltype(A)
            return new{eltype(A), typeof(A), typeof(B), typeof(init)}(A, B, init)
        elseif eltype(init) == eltype(A)
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
    BB      = B*B'
    Sigma   = find_stationary_variance(A, BB)
    if Sigma == zeros(T, n, n)
        return LinearDiffusionStateModel(A, B, mu)
    else
        return LinearDiffusionStateModel(A, B, Distributions.MvNormal(mu, Sigma))
    end
end
        
        
function find_stationary_variance(A::AbstractMatrix{T}, BB::AbstractMatrix{T}) where T
    n       = size(A, 1)
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
initialize(model::LinearDiffusionStateModel{T, F1, F2, TI}) where {T, F1, F2, TI<:Distributions.Sampleable}      = rand(model.init)
initialize(model::LinearDiffusionStateModel{T, F1, F2, TI}) where TI<:AbstractVector{T} where {T, F1, F2}        = model.init
state_dim(model::LinearDiffusionStateModel)         = size(drift(model), 1)
noise_dim(model::LinearDiffusionStateModel)         = size(diffusion(model), 2)
            
            
            
            
            
function drift_function(state_model::LinearDiffusionStateModel)
    A = drift(state_model)
    f(x) = A*x
    return f
end

function diffusion_function(state_model::LinearDiffusionStateModel)
    B = diffusion(state_model)
    g(x) = B
    return g
end
    
    
    
    

function Base.show(io::IO, ::MIME"text/plain", model::LinearDiffusionStateModel)
    print(io, "Linear diffusion process model for the hidden state
    type of hidden state:                   ", state_dim(model),"-dimensional vector
    number of independent Brownian motions: ", noise_dim(model),"
    initial condition:                      ", model.init isa Distributions.Sampleable ? "random" : "fixed")
end

function Base.show(io::IO, model::LinearDiffusionStateModel)
    print(io, state_dim(model),"-dimensional linear diffusion with ", noise_dim(model), "-dimensional Brownian motion")
end

function Base.show(io::IO, ::MIME"text/plain", model::Type{LinearDiffusionStateModel{S1, S2, S3, S4}}) where {S1, S2, S3, S4}
    print(io, "LinearDiffusionStateModel{", S1, ",", S2, ",", S3, ",", S4, "}")
end

function Base.show(io::IO, model::Type{LinearDiffusionStateModel{S1, S2, S3, S4}}) where {S1, S2, S3, S4}
    print(io, "LinearDiffusionStateModel")
end

    
    
    
######################
### TIME EVOLUTION ###
######################
    
    
    
    
    
    
    
(model::LinearDiffusionStateModel)(x::AbstractVector, dt) = x + dt*drift(model)*x + sqrt(dt)*diffusion(model)*randn(eltype(x), noise_dim(model))
(model::LinearDiffusionStateModel)(x::AbstractMatrix, dt) = x + dt*drift(model)*x + sqrt(dt)*diffusion(model)*randn(eltype(x), noise_dim(model), size(x, 2))
    
    
    
    
    
    
    
    
function propagate!(x::AbstractVector, model::LinearDiffusionStateModel, dt)
    x .+= dt .* drift(model) * x + sqrt(dt) .* diffusion(model) * randn(eltype(x), noise_dim(model))
    return x
end 

function propagate!(x::AbstractMatrix, model::LinearDiffusionStateModel, dt)
    x .+= dt .* drift(model) * x + sqrt(dt) .* diffusion(model) * randn(eltype(x), noise_dim(model), size(x, 2))
    return x
end 
            
            
            
            
            
# convert linear model into general model
DiffusionStateModel(model::LinearDiffusionStateModel) = DiffusionStateModel(drift_function(model), diffusion_function(model), initial_condition(model))