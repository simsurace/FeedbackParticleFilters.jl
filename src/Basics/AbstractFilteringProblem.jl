"""
    AbstractFilteringProblem{S1, S2, T1<:TimeType, T2<:TimeType}

Abstract type for a filtering problem for observations of type `S2` in `T2` and hidden states of type `S1` in `T1`.
"""
abstract type AbstractFilteringProblem{S1, S2, T1<:TimeType, T2<:TimeType} end




"""
    state_model(problem::AbstractFilteringProblem)

Return the hidden state model underlying `problem`.
"""
function state_model(problem::AbstractFilteringProblem) end




"""
    obs_model(problem::AbstractFilteringProblem)

Return the observation model underlying `problem`.
"""
function obs_model(problem::AbstractFilteringProblem) end




"""
    state_dim(problem::AbstractFilteringProblem)

Return the dimensionality of the hidden state in `problem`.
"""
state_dim(problem::AbstractFilteringProblem) = state_dim(state_model(problem))




"""
    obs_dim(problem::AbstractFilteringProblem)

Return the dimensionality of the observed state in `problem`.
"""
obs_dim(problem::AbstractFilteringProblem) = obs_dim(obs_model(problem))




state_type(problem::AbstractFilteringProblem{S1, S2, T1, T2}) where {S1, S2, T1, T2}       = S1
obs_type(problem::AbstractFilteringProblem{S1, S2, T1, T2}) where {S1, S2, T1, T2}         = S2
hidden_time_type(problem::AbstractFilteringProblem{S1, S2, T1, T2}) where {S1, S2, T1, T2} = T1
obs_time_type(problem::AbstractFilteringProblem{S1, S2, T1, T2}) where {S1, S2, T1, T2}    = T2




# How to add new filtering problem:
# * Add a struct which is a subtype of AbstractFilteringProblem{S1, S2, T1, T2}, where S1 and S2 are the types of hidden states and observations respectively, and T1 and T2 are the time types of the hidden and observed processes, respectively.
# * Implement methods for 
#   - state_model
#   - obs_model 