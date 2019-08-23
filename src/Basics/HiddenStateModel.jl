"""
    HiddenStateModel{S, T<:TimeType} <: AbstractModel{S}

Abstract type for any model of the hidden state of type `S`.
"""
abstract type HiddenStateModel{S, T} <: AbstractModel{S} end




"""
    state_dim(model::HiddenStateModel)

Returns the dimensionality of the hidden state in `model`.
"""
function state_dim(model::HiddenStateModel) end




"""
    initial_condition(model::HiddenStateModel)

Returns the specification of the initial condition in `model`.
This is either a fixed value or a samplable distribution.
"""
function initial_condition(model::HiddenStateModel) end



"""
    state_type(model::HiddenStateModel)

Returns the data type of the hidden state in `model`.
"""
state_type(model::HiddenStateModel{S,T}) where {S,T} = S




"""
    time_type(model::HiddenStateModel)

Returns the time type of `model`, e.g. `DiscreteTime' or `ContinuousTime'.
"""
time_type(model::HiddenStateModel{S,T}) where {S,T} = T




"""
    initialize(model::HiddenStateModel)

Returns a sample from the initial distribution of `model`.
"""
function initialize(state_model::HiddenStateModel{S, T})::S where {S, T}
    init = initial_condition(state_model)
    if init isa Distributions.Sampleable
        return rand(init)
    else
        return init
    end
end  
    
    
    
    
    
    
    
"""
    propagate!(state[s], model::HiddenStateModel[, dt])

Propagates the state(s) according to the model.
For `ContinuousTime' models, a time step `dt' has to be provided.
Multiple states are given as a matrix with columns corresponding to states, and are processed i.i.d.
"""
function propagate!(state_model::HiddenStateModel{S, T})::S where {S, T} end 