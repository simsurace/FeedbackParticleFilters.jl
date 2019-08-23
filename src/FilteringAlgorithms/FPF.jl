struct FPFState{T, TF} <: AbstractFilterState
    ensemble::UnweightedParticleEnsemble{T}
    eq::PoissonEquation{T, TF}
    function FPFState(ens::UnweightedParticleEnsemble{T}, eq::PoissonEquation{T, TF}) where {T, TF}
        update!(eq, ens)
        new{T, TF}(ens, eq)
    end
end


struct FPF{TS<:HiddenStateModel, TO<:ObservationModel, T<:GainEstimationMethod} <: AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}
    state_model::TS
    obs_model::TO
    gain_method::T
    N::Int
    function FPF(st_mod::HiddenStateModel{S1, ContinuousTime}, ob_mod::ObservationModel{S1, S2, ContinuousTime}, gain_method::GainEstimationMethod, N::Int) where {S1, S2}
        if ob_mod isa DiffusionObservationModel || ob_mod isa LinearDiffusionObservationModel
            return new{typeof(st_mod), typeof(ob_mod), typeof(gain_method)}(st_mod, ob_mod, gain_method, N)
        else
            error("Error: the FPF algorithm is not defined for non-diffusion-type observations. Try ppFPF for counting process observations.")
        end
    end
end



    
    
 
    
#####################
###### METHODS ######
#####################  
    
initial_condition(fpf::FPF) = fpf.state_model.init
no_of_particles(fpf::FPF) = fpf.N
no_of_particles(st::FPFState) = no_of_particles(st.ensemble)
state_model(fpf::FPF) = fpf.state_model
obs_model(fpf::FPF) = fpf.obs_model
gain_estimation_method(fpf::FPF) = fpf.gain_method

mean(st::FPFState) = mean(st.ensemble)
cov(st::FPFState)  = cov(st.ensemble)
var(st::FPFState)  = var(st.ensemble)    
    
function Base.show(io::IO, ::MIME"text/plain", fpf::FPF)
    print(io, "Feedback particle filter algorithm
    hidden state model:                     ", state_model(fpf),"
    observation model:                      ", obs_model(fpf),"
    ensemble size:                          ", no_of_particles(fpf),"
    gain estimation method:                 ", gain_estimation_method(fpf))
end

function Base.show(io::IO, fpf::FPF)
    print(io, "FPF with ", no_of_particles(fpf)," particles and ", gain_estimation_method(fpf))
end




######################
### MAIN ALGORITHM ###
######################
    
function initialize(fpf::FPF)
    return FPFState(fpf)
end    

function update!(filter_state::FPFState, filt_algo::FPF, dY, dt)
    propagate!(filter_state, filt_algo, dt)
    assimilate!(dY, filter_state, filt_algo, dt)
end

function propagate!(filter_state::FPFState, filt_algo::FPF, dt)
    propagate!(filter_state.ensemble, filt_algo.state_model, dt)
end

function assimilate!(dY, filter_state::FPFState, filt_algo::FPF, dt)
    ensemble  = filter_state.ensemble
    eq        = filter_state.eq
    method    = filt_algo.gain_method
    
    error     = dY .- dt * eq.H/2 .- dt * eq.mean_H/2
    solve!(eq, method)
    
    heun!(eq, ensemble, error, method) # stochastic Heun method: intermediate step
        
    applygain!(ensemble, eq, error)
    update!(eq, ensemble)
    return ensemble.positions
end



    
    
    
    
    
      
########################
### HELPER FUNCTIONS ###
########################
    
function heun!(eq::PoissonEquation, ensemble::UnweightedParticleEnsemble, error::AbstractMatrix, method::GainEstimationMethod)
    ensemble2 = deepcopy(ensemble)    
    applygain!(ensemble2, eq, error)
    eq2       = deepcopy(eq)
    update!(eq2, ensemble2)
    solve!(eq2, method)
    eq.gain  .= (eq.gain + eq2.gain) / 2
end    

function gainxerror(gain::Array{T, 3}, error::Array{T, 2}) where T
    out = zeros(T, size(gain, 1), size(gain, 2))
    add_gainxerror!(out, gain, error)
    return out
end

function add_gainxerror!(out::Array{T, 2}, gain::Array{T, 3}, error::Array{T, 2}) where T
    size(gain, 2) == size(error, 2) == size(out, 2) ? N = size(gain, 2) : throw(DimensionMismatch("The provided gain, error, and output array are for different numbers of particles."))
    size(gain, 3) == size(error, 1)                 ? m = size(gain, 3) : throw(DimensionMismatch("The provided gain and error are for different numbers of observed variables."))
    size(gain, 1) == size(out, 1)                   ? n = size(gain, 1) : throw(DimensionMismatch("The provided gain and output array are for different numbers of hidden variables."))
    @inbounds for k in 1:N, i in 1:n, j in 1:m
            out[i, k] += gain[i, k, j] * error[j, k]
    end
    return out
end

function applygain!(ens::UnweightedParticleEnsemble, eq::GainEquation, error::AbstractMatrix)
    add_gainxerror!(ens.positions, eq.gain, error)
end













################################
### CONVENIENCE CONSTRUCTORS ###
################################
    
function FPFState(fpf::FPF)
    N   = no_of_particles(fpf)
    ens = UnweightedParticleEnsemble(hcat([initialize(fpf.state_model) for i in 1:N]...))
    eq  = PoissonEquation(observation_function(fpf.obs_model), ens)
    return FPFState(ens, eq)
end    
        
function FPFState(filt_prob::AbstractFilteringProblem, N)
    return FPFState(state_model(filt_prob), obs_model(filt_prob), N)
end

function FPFState(state_model::HiddenStateModel, obs_model::ObservationModel, N)
    ens = UnweightedParticleEnsemble(hcat([initialize(state_model) for i in 1:N]...))
    eq  = PoissonEquation(observation_function(obs_model), ens)
    return FPFState(ens, eq)
end

function FPF(filt_prob::AbstractFilteringProblem, method::GainEstimationMethod, N)
    if state_model(filt_prob) isa DiffusionStateModel
        st_mod = state_model(filt_prob)
    elseif state_model(filt_prob) isa LinearDiffusionStateModel
        st_mod = state_model(filt_prob)     # convert linear diffusion model into generic one
    else
        error("Error: cannot build feedback particle filter for given state model. Must be a diffusion.")
    end
            
    if obs_model(filt_prob) isa DiffusionObservationModel
        ob_mod = obs_model(filt_prob)
    elseif obs_model(filt_prob) isa LinearDiffusionObservationModel
        ob_mod = obs_model(filt_prob)       # convert linear diffusion model into generic one
    else
        error("Error: cannot build feedback particle filter for given observation model. Must be a diffusion.")
    end
    
    return FPF(st_mod, ob_mod, method, N)
end