abstract type AbstractFilteringAlgorithm{T1<:TimeType, T2<:TimeType} end
abstract type AbstractFilterState end








"""
    update!(filter_state, filter_algo, obs, dt) --> filter_state

Updates the filter state by performing one forward step of the model and then assimilating the observation. 
"""
function update!(
    filter_state::AbstractFilterState, 
    filter_algo::AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}, 
    obs,
    dt
) end