struct DiffusionObservationModel{S1, S2, TF} <: ObservationModel{Vector{S1}, Vector{S2}, ContinuousTime}
    n::Int
    m::Int
    observation_function::TF
end





#####################
### BASIC METHODS ###
#####################
                
                
                
                
                
            
                
state_dim(model::DiffusionObservationModel) = model.n
obs_dim(model::DiffusionObservationModel) = model.m
noise_dim(model::DiffusionObservationModel) = model.m







######################
### TIME EVOLUTION ###
######################







function (model::DiffusionObservationModel{S1, S2, TF})(x::AbstractVector{S1}, dt) where {S1, S2, TF}
    dV = randn(T, noise_dim(model))
    return h(x) * dt + dV * sqrt(dt)
end






function (model::LinearDiffusionObservationModel)(x::AbstractMatrix{T}, dt) where T
    dV = randn(T, noise_dim(model), size(x, 2))
    return mapslices(h, x, dims=1) * dt + dV * sqrt(dt)
end






function (model::DiffusionObservationModel{S1, S2, TF})(x::AbstractVector{Vector{S1}}, dt) where {S1, S2, TF}
    return [model(y, dt) for y in x]
end