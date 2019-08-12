"""
    AbstractModel{S}

Abstract type for any model over states of type `S`.
"""
abstract type AbstractModel{S} end




"""
    state_type(arg)

Returns the type of the hidden state in `arg`. 
Supported argument types:
* Any subtype of AbstractModel
* Any subtype of AbstractFilteringProblem
"""
function state_type(arg) end

"""
    obs_type(arg)

Returns the type of the observed state in `arg`. 
Supported argument types:
* Any subtype of ObservationModel
* Any subtype of AbstractFilteringProblem
"""
function obs_type(arg) end

"""
    time_type(model::AbstractModel)

Returns the time type of `model`. 
"""
function time_type(model::AbstractModel) end