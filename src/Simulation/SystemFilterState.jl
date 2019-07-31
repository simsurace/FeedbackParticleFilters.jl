struct SystemFilterState{S1, S2, TF<:AbstractFilterState}
    hidden_state::S1
    obs::S2
    filter_state::TF
end





"""
    simulate!(filtering_algorithm, filtering_problem, no_of_timesteps, dt)

Runs a simulation of the hidden state, observation, and filtering algorithm for a duration of `no_of_timesteps`.
"""
function SystemFilterState(filt_prob, filt_algo)
    hidden_state = initialize(state_model(filt_prob))
    obs          = obs_model(filt_prob)(dt)
    filter_state = initialize(filt_algo)
    return SystemFilterState(hidden_state, obs, filter_state)
end









"""
    propagate!(sfs, filtering_problem, filtering_algorithm; dt) --> sfs

Propagates the system and filter states for one time-step according to the specified filtering problem and algorithm.
"""
function propagate!(
    sfs::SystemFilterState, 
    filt_prob::AbstractFilteringProblem{S1, S2, ContinuousTime, ContinuousTime}, 
    filt_algo::AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}, 
    dt)
    
    hidden_state = sfs.hidden_state
    obs          = sfs.obs
    filter_state = sfs.filter_state
    state_model  = filt_prob.state_model
    obs_model    = filt_prob.obs_model
    
    propagate!(hidden_state, state_model, dt)
    emit!(obs, hidden_state, obs_model, dt)
    update!(filter_state, filt_algo, obs, dt)
end