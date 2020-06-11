@doc raw"""
    BrownianMotionTorus(n::Int)

Returns a hidden state model corresponding to a Brownian motion on an `n`-dimensional torus, with initial condition drawn uniformly at random.
"""
struct BrownianMotionTorus <: HiddenStateModel{Vector{Float64}, ContinuousTime}
    n::Int
end
                
                
                
     

                
                
                
                
                
#####################
### BASIC METHODS ###
#####################
            
initial_condition(model::BrownianMotionTorus) = Uniform(0,2*pi)
                        
state_dim(model::BrownianMotionTorus) = model.n
                   
                        
        
"""
    noise_dim(model)

Returns the dimension of the Brownian motion ``W_t`` in the diffusion model ``dX_t = f(X_t)dt + g(X_t)dW_t``.
"""                        
noise_dim(model::BrownianMotionTorus) = model.n

function initialize(model::BrownianMotionTorus)
    x = rand(initial_condition(model), state_dim(model))
    return x
end
                
   
                            
                
function Base.show(io::IO, ::MIME"text/plain", model::BrownianMotionTorus)
    print(io, "Brownian motion on the ", state_dim(model), "-torus for the hidden state
    type of hidden state:                   ", state_dim(model),"-dimensional vector
    number of independent Brownian motions: ", noise_dim(model),"
    initial condition:                      uniform")
end

function Base.show(io::IO, model::BrownianMotionTorus)
    print(io, "Brownian motion on the ", state_dim(model), "-torus")
end
  
                
                
                
                
                
                
                
                
 



######################
### TIME EVOLUTION ###
######################


                
                
                
                
                
(model::BrownianMotionTorus)(x::AbstractVector, dt) = mod2pi(x + sqrt(dt)*randn(eltype(x), noise_dim(model)))


                
                
                
                
                
                
                
function (model::BrownianMotionTorus)(x::AbstractMatrix, dt)
    N   = size(x, 2)
    F   = drift_function(model)(x)
    G   = diffusion_function(model)(x)
    if ndims(G) == 2
        G = repeat(G, 1, 1, N)
    end
    out = zeros(eltype(x), state_dim(model), N)
    sqr = sqrt(dt)
    @inbounds for b in 1:N
        for a in 1:state_dim(model)
            out[a,b]  +=  mod2pi( x[a,b]  +  sqr * randn(eltype(x)) )
        end
    end
    return out
end 
                
                
                
                
                
                
                
function propagate!(x::AbstractVector, model::BrownianMotionTorus, dt)
    N   = size(x, 2)
    sqr = sqrt(dt)
    @inbounds for a in 1:state_dim(model)
            x[a]  +=  sqr * randn(eltype(x))
            x[a]   =  mod2pi(x[a])
    end
    return x
end 

                
                
                
                
                
                
                
                
                
function propagate!(x::AbstractMatrix, model::BrownianMotionTorus, dt)
    N   = size(x, 2)
    sqr = sqrt(dt)
    @inbounds for b in 1:N
        for a in 1:state_dim(model)
                x[a,b]  +=  sqr * randn(eltype(x))
                x[a,b]   =  mod2pi(x[a,b])
        end
    end
    return x
end 











################################
### CONVENIENCE CONSTRUCTORS ###
################################

const BrownianMotionCircle = BrownianMotionTorus(1)