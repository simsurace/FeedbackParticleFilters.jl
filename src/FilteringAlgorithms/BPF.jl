struct BPFState{T} <: AbstractFilterState
    ensemble::WeightedParticleEnsemble{T}
    function BPFState(ens::WeightedParticleEnsemble{T}) where T
        new{T}(ens)
    end
end


struct BPF{TS<:HiddenStateModel, TO<:ObservationModel, T<:Real} <: AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}
    state_model::TS
    obs_model::TO
    N::Int
    alpha::T
end



    
    
 
    
#####################
###### METHODS ######
#####################  
    
initial_condition(filter::BPF) = filter.state_model.init
no_of_particles(filter::BPF) = filter.N
no_of_particles(st::BPFState) = no_of_particles(st.ensemble)
state_model(filter::BPF) = filter.state_model
obs_model(filter::BPF) = filter.obs_model

mean(st::BPFState) = mean(st.ensemble)
cov(st::BPFState)  = cov(st.ensemble)
var(st::BPFState)  = var(st.ensemble)    
    
function Base.show(io::IO, ::MIME"text/plain", filter::BPF)
    print(io, "Bootstrap particle filter algorithm
    hidden state model:                              ", state_model(filter),"
    observation model:                               ", obs_model(filter),"
    ensemble size:                                   ", no_of_particles(filter),"
    threshold for effective number of particles (%): ", 100*filter.alpha)
end

function Base.show(io::IO, filter::BPF)
    print(io, "BPF with ", no_of_particles(filter)," particles ")
end




######################
### MAIN ALGORITHM ###
######################
    
function initialize(filter::BPF)
    return BPFState(filter)
end    

function propagate!(filter_state::BPFState, filt_algo::BPF, dt)
    propagate!(filter_state.ensemble, state_model(filt_algo), dt)
end

function assimilate!(observation, filter_state::BPFState, filt_algo::BPF, dt)
    ensemble  = filter_state.ensemble
    
    update_weights!(ensemble, obs_model(filt_algo), observation, dt)

    if eff_no_of_particles(ensemble) < filt_algo.alpha * no_of_particles(ensemble)
        resample!(ensemble)
    end

    return ensemble.positions
end

function update!(filter_state::BPFState, filt_algo::BPF, dY, dt)
    propagate!(filter_state, filt_algo, dt)
    assimilate!(dY, filter_state, filt_algo, dt)
end



    
    
    
    
    
      



########################
### HELPER FUNCTIONS ###
########################

function update_weights!(ensemble::WeightedParticleEnsemble, model::DiffusionObservationModel, dY, dt)
    H         = mapslices(observation_function(model), ensemble.positions, dims=1)
    mean_H    = StatsBase.mean(H, ensemble.weights, dims=2)
    
    @inbounds for i in 1:no_of_particles(ensemble), j in 1:obs_dim(model)
        ensemble.weights[i] += ensemble.weights[i] * ( H[j,i] - mean_H[j,1] ) * (dY[j] - mean_H[j,1] * dt)
    end
end

function update_weights!(ensemble::WeightedParticleEnsemble, model::CountingObservationModel, dN, dt)
    H         = mapslices(observation_function(model), ensemble.positions, dims=1)
    mean_H    = StatsBase.mean(H, ensemble.weights, dims=2)
    
    @inbounds for i in 1:no_of_particles(ensemble), j in 1:obs_dim(model)
        ensemble.weights[i] += ensemble.weights[i] * ( H[j,i] - mean_H[j,1] ) * (dN[j] - mean_H[j,1] * dt) / mean_H[j,1]
    end
end









################################
### CONVENIENCE CONSTRUCTORS ###
################################
    
function BPFState(filter::BPF)
    N   = no_of_particles(filter)
    ens = WeightedParticleEnsemble(hcat([initialize(filter.state_model) for i in 1:N]...), StatsBase.ProbabilityWeights(fill(1/N, N)))
    return BPFState(ens)
end    
        
function BPFState(filt_prob::AbstractFilteringProblem, N)
    return BPFState(state_model(filt_prob), obs_model(filt_prob), N)
end

function BPFState(state_model::HiddenStateModel, obs_model::ObservationModel, N)
    ens = WeightedParticleEnsemble(hcat([initialize(state_model) for i in 1:N]...), StatsBase.ProbabilityWeights(fill(1/N, N)))
    return BPFState(ens)
end

function BPF(filt_prob::AbstractFilteringProblem, N, alpha)
    st_mod = state_model(filt_prob)
    ob_mod = obs_model(filt_prob)
    
    return BPF(st_mod, ob_mod, N, alpha)
end