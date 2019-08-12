struct DiffusionObservationModel{S1, S2, TF} <: ObservationModel{Vector{S1}, Vector{S2}, ContinuousTime}
    n::Int
    m::Int
    observation_function::TF
end





#####################
### BASIC METHODS ###
#####################
                
                
                
                
                
            
                
drift(model::DiffusionObservationModel) = model.observation_function
state_dim(model::DiffusionObservationModel) = model.n
obs_dim(model::DiffusionObservationModel) = model.m
noise_dim(model::DiffusionObservationModel) = model.m






observation_function(obs_model::DiffusionObservationModel) = drift(obs_model)









function Base.show(io::IO, ::MIME"text/plain", model::DiffusionObservationModel)
    print(io, "Diffusion process model for the observation
    type of hidden state:                   ", state_dim(model),"-dimensional vector
    type of observation:                    ", obs_dim(model),"-dimensional vector
    number of independent Brownian motions: ", noise_dim(model))
end

function Base.show(io::IO, model::DiffusionObservationModel)
    print(io, obs_dim(model),"-dimensional diffusion with ", noise_dim(model), "-dimensional Brownian motion")
end

function Base.show(io::IO, ::MIME"text/plain", model::Type{DiffusionObservationModel{S,T}}) where {S,T}
    print(io, "DiffusionObservationModel{", S, ",", T, "}")
end

function Base.show(io::IO, model::Type{DiffusionObservationModel{S,T}}) where {S,T}
    print(io, "DiffusionObservationModel")
end


######################
### TIME EVOLUTION ###
######################







function (model::DiffusionObservationModel{S1, S2, TF})(x::AbstractVector{S1}, dt) where {S1, S2, TF}
    dV = randn(S1, noise_dim(model))
    return drift(model)(x) * dt + dV * sqrt(dt)
end






function (model::DiffusionObservationModel)(x::AbstractMatrix{T}, dt) where T
    dV = randn(T, noise_dim(model), size(x, 2))
    return mapslices(h, x, dims=1) * dt + dV * sqrt(dt)
end






function (model::DiffusionObservationModel{S1, S2, TF})(x::AbstractVector{Vector{S1}}, dt) where {S1, S2, TF}
    return [model(y, dt) for y in x]
end







################################
### CONVENIENCE CONSTRUCTORS ###
################################
        
function ScalarDiffusionObservationModel(h::Function, n = 1)
    if n == 1
        H1(x) = [h(x[1])]
        return DiffusionObservationModel{Float64, Float64, typeof(H1)}(n, 1, H1)
    elseif n > 1
        H2(x) = [h(x)]
        return DiffusionObservationModel{Float64, Float64, typeof(H2)}(n, 1, H2)
    end
    
end