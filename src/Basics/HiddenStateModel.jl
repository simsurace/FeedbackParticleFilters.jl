"""
    HiddenStateModel{S, T<:TimeType} <: AbstractModel{S}

Abstract type for any model of the hidden state of type `S`.
"""
abstract type HiddenStateModel{S, T} <: AbstractModel{S} end




function state_dim(model::HiddenStateModel) end









function initial_condition(model::HiddenStateModel) end









state_type(model::HiddenStateModel{S,T}) where {S,T} = S











time_type(model::HiddenStateModel{S,T}) where {S,T} = T







function initialize(state_model::HiddenStateModel{S, T})::S where {S, T}
    init = initial_condition(state_model)
    if init isa Distributions.Sampleable
        return rand(init)
    else
        return init
    end
end  