struct LinearDiffusionObservationModel{S, F} <: ObservationModel{Vector{S}, Vector{S}, ContinuousTime}
    observation_matrix::F
    LinearDiffusionObservationModel(C::AbstractMatrix{S}) where S = new{S, typeof(C)}(C)
end






#####################
### BASIC METHODS ###
#####################
                
                
                
                
                
            
   
drift(model::LinearDiffusionObservationModel)     = model.observation_matrix
state_dim(model::LinearDiffusionObservationModel) = size(drift(model), 2)
obs_dim(model::LinearDiffusionObservationModel)   = size(drift(model), 1)
noise_dim(model::LinearDiffusionObservationModel) = obs_dim(model)






function observation_function(obs_model::LinearDiffusionObservationModel)
    C = drift(obs_model)
    h(x) = C*x
    return h
end





function Base.show(io::IO, ::MIME"text/plain", model::LinearDiffusionObservationModel)
    print(io, "Linear diffusion process model for the observation
    type of hidden state:                   ", state_dim(model),"-dimensional vector
    type of observation:                    ", obs_dim(model),"-dimensional vector
    number of independent Brownian motions: ", noise_dim(model))
end

function Base.show(io::IO, model::LinearDiffusionObservationModel)
    print(io, obs_dim(model),"-dimensional linear diffusion with ", noise_dim(model), "-dimensional Brownian motion")
end

function Base.show(io::IO, ::MIME"text/plain", model::Type{LinearDiffusionObservationModel{S,T}}) where {S,T}
    print(io, "LinearDiffusionObservationModel{", S, ",", T, "}")
end

function Base.show(io::IO, model::Type{LinearDiffusionObservationModel{S,T}}) where {S,T}
    print(io, "LinearDiffusionObservationModel")
end
    











######################
### TIME EVOLUTION ###
######################







function (model::LinearDiffusionObservationModel)(x::AbstractVector{T}, dt) where T
    dV = randn(T, noise_dim(model))
    return drift(model) * x * dt + dV * sqrt(dt)
end
    
function (model::LinearDiffusionObservationModel)(x::AbstractMatrix{T}, dt) where T
    dV = randn(T, noise_dim(model), size(x, 2))
    return drift(model) * x * dt + dV * sqrt(dt)
end











# convert linear model into general model
function DiffusionObservationModel(ob_mod::LinearDiffusionObservationModel) 
    h = observation_function(ob_mod)
    return DiffusionObservationModel{eltype(state_type(ob_mod)), eltype(obs_type(ob_mod)), typeof(h)}(state_dim(ob_mod), obs_dim(ob_mod), h)
end