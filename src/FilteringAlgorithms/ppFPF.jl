struct ppFPFState{T, TF1, TF2} <: AbstractFilterState
    ensemble::UnweightedParticleEnsemble{T}
    eq_dN::PoissonEquation{T, TF1}
    eq_dt::PoissonEquation{T, TF2}
    function ppFPFState(ens::UnweightedParticleEnsemble{T}, eq_dN::PoissonEquation{T, TF1}, eq_dt::PoissonEquation{T, TF2}) where {T, TF1, TF2}
        update!(eq_dN, ens)
        update!(eq_dt, ens)
        new{T, TF1, TF2}(ens, eq_dN, eq_dt)
    end
end


struct ppFPF{TS<:HiddenStateModel, TO<:ObservationModel, T1<:GainEstimationMethod, T2<:ParticleFlowMethod} <: AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}
    state_model::TS
    obs_model::TO
    gain_method::T1
    flow_method::T2
    N::Int
    function ppFPF(st_mod::HiddenStateModel{S1, ContinuousTime}, ob_mod::ObservationModel{S1, S2, ContinuousTime}, gain_method::GainEstimationMethod, flow_method::ParticleFlowMethod, N::Int) where {S1, S2}
        if ob_mod isa CountingObservationModel
            return new{typeof(st_mod), typeof(ob_mod), typeof(gain_method), typeof(flow_method)}(st_mod, ob_mod, gain_method, flow_method, N)
        else
            error("Error: the ppFPF algorithm is only defined for counting process observations.")
        end
    end
end
    
    
    
    
    
#####################
###### METHODS ######
#####################  
    
initial_condition(filter::ppFPF) = filter.state_model.init
no_of_particles(filter::ppFPF) = filter.N
no_of_particles(st::ppFPFState) = no_of_particles(st.ensemble)
state_model(filter::ppFPF) = filter.state_model
obs_model(filter::ppFPF) = filter.obs_model
gain_estimation_method(filter::ppFPF) = filter.gain_method
flow_method(filter::ppFPF) = filter.flow_method

mean(st::ppFPFState) = mean(st.ensemble)
cov(st::ppFPFState)  = cov(st.ensemble)
var(st::ppFPFState)  = var(st.ensemble)    
    
function Base.show(io::IO, ::MIME"text/plain", filter::ppFPF)
    print(io, "Point-process feedback particle filter algorithm
    hidden state model:                     ", state_model(filter),"
    observation model:                      ", obs_model(filter),"
    ensemble size:                          ", no_of_particles(filter),"
    gain estimation method:                 ", gain_estimation_method(filter),"
    particle flow method:                   ", flow_method(filter))
end

function Base.show(io::IO, filter::ppFPF)
    print(io, "ppFPF with ", no_of_particles(filter)," particles,", gain_estimation_method(filter), ", and ", flow_method(filter))
end




######################
### MAIN ALGORITHM ###
######################
    
function initialize(filter::ppFPF)
    return ppFPFState(filter)
end    

function update!(filter_state::ppFPFState, filt_algo::ppFPF, dY, dt)
    propagate!(filter_state, filt_algo, dt)
    assimilate!(dY, filter_state, filt_algo, dt)
end

function propagate!(filter_state::ppFPFState, filt_algo::ppFPF, dt)
    propagate!(filter_state.ensemble, filt_algo.state_model, dt)
end

function assimilate!(dN, filter_state::ppFPFState, filt_algo::ppFPF, dt)
    ensemble  = filter_state.ensemble
    eq_dt     = filter_state.eq_dt
    eq_dN     = filter_state.eq_dN
    method    = gain_estimation_method(filt_algo)
    fl_method = flow_method(filt_algo)
    
    # dt update
    solve!(eq_dt, method)

    ensemble.positions .+= dt * view(eq_dt.gain, :, :, 1)
        
    # event update
    logl(x) = LinearAlgebra.dot(eq_dN.h(x), dN)
    flow!(ensemble, logl, fl_method)
        
    update!(eq_dt, ensemble)
    return ensemble.positions
end



    
    
    
    
    
      




################################
### CONVENIENCE CONSTRUCTORS ###
################################
    
function ppFPFState(filter::ppFPF)
    N = no_of_particles(filter)
    return ppFPFState(state_model(filter), obs_model(filter), N)
end    
        
function ppFPFState(filt_prob::AbstractFilteringProblem, N)
    return ppFPFState(state_model(filt_prob), obs_model(filt_prob), N)
end

function ppFPFState(state_model::HiddenStateModel, obs_model::ObservationModel, N)
    ens    = UnweightedParticleEnsemble(hcat([initialize(state_model) for i in 1:N]...))
    eq_dt  = PoissonEquation(x -> -sum(observation_function(obs_model)(x)), ens)
    eq_dN  = PoissonEquation(x -> log.(observation_function(obs_model)(x)), ens)
    return ppFPFState(ens, eq_dN, eq_dt)
end

function ppFPF(filt_prob::AbstractFilteringProblem, method::GainEstimationMethod, fl_method::ParticleFlowMethod, N)
    st_mod = state_model(filt_prob)
  
    if obs_model(filt_prob) isa CountingObservationModel
        ob_mod = obs_model(filt_prob)
    else
        error("Error: the EKSPF algorithm is only defined for counting process observations.")
    end
    
    return ppFPF(st_mod, ob_mod, method, fl_method, N)
end