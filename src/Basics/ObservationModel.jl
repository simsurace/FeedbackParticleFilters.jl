"""
    ObservationModel{S1, S2, T} <: AbstractModel{S2}

Abstract type for any model of observations of type `S2` and hidden states of type `S1`.

Calling the model samples a new observation. For discrete-time or static models, 
    
    model(x)

returns a new observation based on hidden state `x`.

For continuous-time models,

    model(x, dt)

returns a new observation based on hidden state `x` and time-step `dt`.

If `x` is a vector of `state_type` elements or a batch matrix, an observation is generated for each element of the batch.
"""
abstract type ObservationModel{S1, S2, T} <: AbstractModel{S2} end












function state_dim(model::ObservationModel) end
function obs_dim(model::ObservationModel) end









state_type(model::ObservationModel{S1, S2, T}) where {S1, S2, T} = S1
obs_type(model::ObservationModel{S1, S2, T}) where {S1, S2, T}   = S2
time_type(model::ObservationModel{S1, S2, T}) where {S1, S2, T}  = T