struct KBF{T1<:AbstractMatrix{T}, T2<:AbstractMatrix{T}, T3<:AbstractMatrix{T}, T4<:AbstractMatrix{T}, TI} <: AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime} 
    A::T1
    BB::T2
    C::T3
    CC::T4
    init::TI
end



struct KBState{TM, TP} <: AbstractFilterState
    gauss::MultivariateGaussian{TM, TP}
end


mean(st::KBState) = mean(st.gauss)
cov(st::KBState)  = cov(st.gauss)
var(st::KBState)  = diag(cov(st))



# convenience constructors


function KBF(A, B, C)
    find_stationary_variance(A, B)
    init = KBState(MultivariateGaussian(zeros(T, n), Sigma))
    return KBF(A, B, C, init)
end

function KBF(filt_prob::AbstractFilteringProblem) 
    st_mod = state_model(filt_prob)
    ob_mod = obs_model(filt_prob)
    return KBF(st_mod, obs_mod)
end

function KBF(state_model::LinearDiffusionStateModel, obs_model::LinearDiffusionObservationModel) 
    A  = drift(state_model)
    B  = diffusion(state_model)
    C  = drift(obs_model)
    init = LDMinit2KBFinit(initial_condition(state_model))
    return KBF(A, B, C, init)
end

function KBF(A, B, C, init)
    BB = B*B'
    CC = C'*C
    return KBF(A, BB, C, CC, init)
end

KBState(mean::AbstractVector{T}, cov::AbstractMatrix{T}) where T = KBState(MultivariateGaussian(mean, cov)) 
KBState(n::Int; T = Float64)                                     = KBState(zeros(T, n), zeros(T, n, n))










# helper functions for constructors above (not exported)


function LDMinit2KBFinit(init) 
    # converts an init specification of LinearDiffusionModel to a corresponding specification of KalmanBucyFilter
    if init isa Distributions.AbstractMvNormal
        mean = init.mu
        cov  = init.sig
        
    else if init isa AbstractVector
        mean = init
        cov  = zeros(eltype(mean), length(mean), length(mean))
    else
        error("Error: invalid initial condition in state_model. Initial condition must be Gaussian.")
    end
    return MultivariateGaussian(mean, cov)
end
        













function initialize(kbf::KBF)
    Sigma = find_stationary_variance(kbf.A, kbf.BB)
    return KBState(MultivariateGaussian(zeros(T, n), Sigma), MultivariateGaussian(zeros(T, n), Sigma))
end




function update!(filter_state::KBState, filter_algo::KBF, dY, dt)
    mean  = mean(filter_state.gauss)
    cov   = cov(filter_state.gauss)
    
    A      = filter_algo.A
    BB     = filter_algo.BB
    C      = filter_algo.C
    CC     = filter_algo.CC
    
    mean .= mean + dt * A * mean + cov * C' * (dY - C * mean * dt)
    cov  .= cov  + dt * (BB - A * cov - cov * A' - cov * CC * cov)
    
    return filter_state
end