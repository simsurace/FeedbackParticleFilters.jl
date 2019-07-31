abstract type AbstractFilteringAlgorithm{T1<:TimeType, T2<:TimeType} end
abstract type AbstractFilterState end








"""
    update!(filter_state, filter_algo, dt)


"""
function update!(
    filter_state::AbstractFilterState, 
    filter_algo::AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}, 
    obs,
    dt
) end
















#(algo::AbstractFilteringAlgorithm{S1, S2, DiscreteTime, DiscreteTime})(filter_state::AbstractFilterState) where {S1, S2}                              = propagate!(filter_state, algo)
#(algo::AbstractFilteringAlgorithm{S1, S2, DiscreteTime, DiscreteTime})(filter_state::AbstractFilterState, obs::S2) where {S1, S2}                     = propagate!(filter_state, algo, obs)
#(algo::AbstractFilteringAlgorithm{S1, S2, ContinuousTime, DiscreteTime})(filter_state::AbstractFilterState, dt::DT) where {S1, S2, DT<:Real}          = propagate!(filter_state, algo, dt)
#(algo::AbstractFilteringAlgorithm{S1, S2, ContinuousTime, DiscreteTime})(filter_state::AbstractFilterState, obs::S2, dt::DT) where {S1, S2, DT<:Real} = propagate!(filter_state, algo, obs, dt)




