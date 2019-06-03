#abstract type StochasticCalculus end
#struct ItoCalculus <: StochasticCalculus end
#struct StratonovichCalculus <: StochasticCalculus end
#using Distributions

"""
    StateModel

Abstract type for the hidden state model of the filtering problem.

Concrete types: DiffusionStateModel, ScalarDiffusionStateModel
"""
abstract type StateModel end

"""
    Propagator(model::StateModel)

Returns a function called `propagate!` that propagates the state model by one time step.
"""
function Propagator(model::StateModel) end

"""
    DiffusionStateModel(drift_function::Function, diffusion_function::Function, n::Int, nprime::Int) <: ObservationModel

A diffusion process hidden state model dX_t = f(X_t)dt + g(X_t)dW_t, where f is the `drift_function`, g is the `observation_function`, X_t is the `n`-dimensional hidden state at time t, and W_t is an `nprime`-dimensional Brownian motion process.
"""
struct DiffusionStateModel <: StateModel
    drift_function::Function
    diffusion_function::Function
    initial_distribution::Union{Array{Float64,1}, Distributions.Sampleable}
    n::Int
    nprime::Int
    #calculus::StochasticCalculus
end

struct ScalarDiffusionStateModel <: StateModel
    drift_function::Function
    diffusion_function::Function
    initial_distribution::Union{Float64, Distributions.Sampleable}
    #calculus::StochasticCalculus
end

function FPFEnsemble(state_model::ScalarDiffusionStateModel, N::Int)
    x0 = state_model.initial_distribution
    if typeof(x0) <: Distributions.Sampleable
        FPFEnsemble(x0, N)
    elseif typeof(x0) == Float64
        FPFEnsemble([x0], N)
    end#if
end
    
function Propagator(state_model::ScalarDiffusionStateModel, dt::Float64)
    function propagate!(x::Float64)
        x + state_model.drift_function(x)*dt + state_model.diffusion_function(x)*sqrt(dt)*randn()
    end
    function propagate!(ensemble::FPFEnsemble{Float64})
        for i in 1:ensemble.size
            ensemble.positions[i] += state_model.drift_function(ensemble.positions[i])*dt + state_model.diffusion_function(ensemble.positions[i])*sqrt(dt)*randn()
        end#for
    end
    return propagate!
end