struct KBState{TG<:MultivariateGaussian} <: AbstractFilterState
    gauss::TG
end







struct KBF{T1, T2, T3, T4, TI<:KBState} <: AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime} 
    A::T1
    BB::T2
    C::T3
    CC::T4
    init::TI
    function KBF(A::AbstractMatrix{S}, BB::AbstractMatrix{S}, C::AbstractMatrix{S}, CC::AbstractMatrix{S}, init::KBState) where {S, TI}
        if size(A, 1) == size(A, 2) == size(BB, 1) == size(BB, 2) == size(CC, 1) == size(C, 2)
            return new{typeof(A), typeof(BB), typeof(C), typeof(CC), typeof(init)}(A, BB, C, CC, init)
        else 
            throw(DimensionMismatch("Input matrices have inconsistent sizes."))
        end
    end
end

    
    
    
    
    
#####################
###### METHODS ######
#####################  
    
initial_condition(kbf::KBF) = kbf.init


mean(st::KBState) = mean(st.gauss)
cov(st::KBState)  = cov(st.gauss)
var(st::KBState)  = var(st.gauss)






        



######################
### MAIN ALGORITHM ###
######################

function initialize(kbf::KBF)
    return KBState(kbf)
end




function update!(filter_state::KBState, filter_algo::KBF, dY, dt) where {TM, TP}
    mean1  = mean(filter_state.gauss)
    cov1   = cov(filter_state.gauss)
    
    A      = filter_algo.A
    BB     = filter_algo.BB
    C      = filter_algo.C
    CC     = filter_algo.CC
    
    mean2  = mean1 + dt * A * mean1 + cov1 * C' * (dY .- C * mean1 * dt)
    cov2   = cov1  + dt * (BB + A * cov1 + cov1 * A' - cov1 * CC * cov1) |> LinearAlgebra.Symmetric
            
    mean1 .= mean2
    cov1  .= cov2
    
    return filter_state
end
    








    
    
    
    

################################
### CONVENIENCE CONSTRUCTORS ###
################################

KBState(mean::AbstractVector{T}, cov::AbstractMatrix{T}) where T = KBState(MultivariateGaussian(mean, cov)) 
KBState(n::Int; T = Float64)                                     = KBState(zeros(T, n), zeros(T, n, n))
KBState(filt_prob::FilteringProblem)                             = KBState(initial_condition(state_model(filt_prob)))
KBState(kbf::KBF)                                                = initial_condition(kbf)

function KBState(init::AbstractVector)
    mean = init
    cov  = zeros(eltype(mean), length(mean), length(mean))
    return KBState(mean, cov)
end
    
function KBState(init::Distributions.Sampleable) 
    if init isa Distributions.AbstractMvNormal
        mean = init.μ
        cov  = Matrix(init.Σ)
    else
        error("Error: invalid initial condition in state_model. Initial condition must be Gaussian.")
    end
    return KBState(mean, cov)
end
        
        
        
function KBF(A, B, C)
    BB = B*B'
    Sigma = find_stationary_variance(A, BB)
    init = KBState(MultivariateGaussian(zeros(eltype(Sigma), size(Sigma, 1)), Sigma))
    CC = C'*C
    return KBF(A, BB, C, CC, init)
end

function KBF(filt_prob::AbstractFilteringProblem) 
    st_mod = state_model(filt_prob)
    ob_mod = obs_model(filt_prob)
    return KBF(st_mod, ob_mod)
end

function KBF(state_model::LinearDiffusionStateModel, obs_model::LinearDiffusionObservationModel) 
    A  = drift(state_model)
    B  = diffusion(state_model)
    C  = drift(obs_model)
    init = KBState(initial_condition(state_model))
    return KBF(A, B, C, init)
end

function KBF(A, B, C, init)
    BB = B*B'
    CC = C'*C
    return KBF(A, BB, C, CC, init)
end