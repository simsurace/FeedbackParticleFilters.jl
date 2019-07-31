"""
    HiddenStateModel{S, T<:TimeType} <: AbstractModel{S}

Abstract type for any model of the hidden state of type `S`.
"""
abstract type HiddenStateModel{S, T} <: AbstractModel{S} end




function state_dim(model::HiddenStateModel) end









state_type(model::HiddenStateModel{S,T}) where {S,T} = S











time_type(model::HiddenStateModel{S,T}) where {S,T} = T









