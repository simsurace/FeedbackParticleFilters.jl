# EKSPF uses FPFState
const EKSPFState = FPFState

struct EKSPF{TS<:HiddenStateModel, TO<:ObservationModel} <: AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}
    state_model::TS
    obs_model::TO
    N::Int
    function EKSPF(st_mod::HiddenStateModel{S1, ContinuousTime}, ob_mod::ObservationModel{S1, S2, ContinuousTime}, N::Int) where {S1, S2}
        if ob_mod isa CountingObservationModel
            return new{typeof(st_mod), typeof(ob_mod)}(st_mod, ob_mod, N)
        else
            error("Error: the EKSPF algorithm is only defined for counting process observations.")
        end
    end
end



    
    
 
    
#####################
###### METHODS ######
#####################  
    
initial_condition(ekspf::EKSPF) = initial_condition(state_model(ekspf))
no_of_particles(ekspf::EKSPF) = ekspf.N
state_model(ekspf::EKSPF) = ekspf.state_model
obs_model(ekspf::EKSPF) = ekspf.obs_model
gain_estimation_method(ekspf::EKSPF) = ConstantGainEKSPF()   
    
function Base.show(io::IO, ::MIME"text/plain", ekspf::EKSPF)
    print(io, "Ensemble Kushner-Stratonovich-Poisson filter algorithm
    hidden state model:                     ", state_model(ekspf),"
    observation model:                      ", obs_model(ekspf),"
    ensemble size:                          ", no_of_particles(ekspf))
end

function Base.show(io::IO, ekspf::EKSPF)
    print(io, "EKSPF with ", no_of_particles(ekspf)," particles")
end




######################
### MAIN ALGORITHM ###
######################
    
function initialize(ekspf::EKSPF)
    return EKSPFState(ekspf)
end    

function update!(filter_state::EKSPFState, filt_algo::EKSPF, dY, dt)
    propagate!(filter_state, filt_algo, dt)
    assimilate!(dY, filter_state, filt_algo, dt)
end

function propagate!(filter_state::EKSPFState, filt_algo::EKSPF, dt)
    propagate!(filter_state.ensemble, filt_algo.state_model, dt)
end

function assimilate!(dN, filter_state::EKSPFState, filt_algo::EKSPF, dt)
    ensemble  = filter_state.ensemble
    eq        = filter_state.eq
    method    = ConstantGainEKSPF()
    
    error     = dN .- dt * eq.H
    solve!(eq, method)
        
    applygain!(ensemble, eq, error)
    update!(eq, ensemble)
    return ensemble.positions
end



    
    
    
    
    
      

        
        
        
################################
### CONVENIENCE CONSTRUCTORS ###
################################
    
function EKSPFState(ekspf::EKSPF)
    N   = no_of_particles(ekspf)
    ens = UnweightedParticleEnsemble(hcat([initialize(ekspf.state_model) for i in 1:N]...))
    eq  = PoissonEquation(observation_function(ekspf.obs_model), ens)
    return EKSPFState(ens, eq)
end     

function EKSPF(filt_prob::AbstractFilteringProblem, N)
    st_mod = state_model(filt_prob)
  
    if obs_model(filt_prob) isa CountingObservationModel
        ob_mod = obs_model(filt_prob)
    else
        error("Error: the EKSPF algorithm is only defined for counting process observations.")
    end
    
    return EKSPF(st_mod, ob_mod, N)
end
