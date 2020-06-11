struct CountingObservationModel{S1, S2, TF} <: ObservationModel{Vector{S1}, Vector{S2}, ContinuousTime}
    n::Int
    m::Int
    observation_function::TF
end





#####################
### BASIC METHODS ###
#####################
                
                
                
                
                
            
                
drift(model::CountingObservationModel) = model.observation_function
state_dim(model::CountingObservationModel) = model.n
obs_dim(model::CountingObservationModel) = model.m
noise_dim(model::CountingObservationModel) = model.m






observation_function(obs_model::CountingObservationModel) = drift(obs_model)









function Base.show(io::IO, ::MIME"text/plain", model::CountingObservationModel)
    print(io, "Counting process model for the observation
    type of hidden state:                    ", state_dim(model),"-dimensional vector
    type of observation:                     ", obs_dim(model),"-dimensional vector
    number of independent Poisson processes: ", noise_dim(model))
end

function Base.show(io::IO, model::CountingObservationModel)
    print(io, obs_dim(model),"-dimensional counting process")
end

function Base.show(io::IO, ::MIME"text/plain", model::Type{CountingObservationModel{S,T}}) where {S,T}
    print(io, "CountingObservationModel{", S, ",", T, "}")
end

function Base.show(io::IO, model::Type{CountingObservationModel{S,T}}) where {S,T}
    print(io, "CountingObservationModel")
end


######################
### TIME EVOLUTION ###
######################







function (model::CountingObservationModel{S1, S2, TF})(x::AbstractVector{S1}, dt) where {S1, S2, TF}
    return [rand(Poisson(r * dt)) for r in observation_function(model)(x)]
end







function (model::CountingObservationModel)(x::AbstractMatrix{T}, dt) where T
    h  = observation_function(model)
    return [rand(Poisson(r * dt)) for r in mapslices(h, x, dims=1)]
end






function (model::CountingObservationModel{S1, S2, TF})(x::AbstractVector{Vector{S1}}, dt) where {S1, S2, TF}
    return [model(y, dt) for y in x]
end