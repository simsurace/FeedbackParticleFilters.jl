"""
    Propagator(model::StateModel)

Returns a function called `propagate!` that propagates the state model by one time step.
"""
function Propagator(model::HiddenStateModel) end

"""
    DiffusionStateModel(drift_function::Function, diffusion_function::Function, n::Int, nprime::Int) <: ObservationModel

A diffusion process hidden state model dX_t = f(X_t)dt + g(X_t)dW_t, where f is the `drift_function`, g is the `observation_function`, X_t is the `n`-dimensional hidden state at time t, and W_t is an `nprime`-dimensional Brownian motion process.
"""
abstract type DiffusionStateModel{S} <: HiddenStateModel{S} end

struct ScalarDiffusionStateModel <: DiffusionStateModel{Float64}
    drift_function::Function
    diffusion_function::Function
    initial_distribution::Union{Float64, Distributions.Sampleable}
    #calculus::StochasticCalculus
    ScalarDiffusionStateModel(f::Function, g::Function, init::Union{Float64, Distributions.Sampleable}) = 
        if hasmethod(f, Tuple{Float64}) && hasmethod(g, Tuple{Float64}) && typeof(f(0.)) == Float64 && typeof(g(0.)) == Float64
            new(f, g, init)
        else
            error("Drift or diffusion function has wrong domain or codomain (must be Float64 in each case).")
        end
end

struct VectorDiffusionStateModel <: DiffusionStateModel{Vector{Float64}}
    drift_function::Function
    diffusion_function::Function
    initial_distribution::Union{Array{Float64,1}, Distributions.Sampleable}
    n::Int
    nprime::Int
    #calculus::StochasticCalculus
    VectorDiffusionStateModel(f::Function, g::Function, init::Union{Float64, Array{Float64,1}, Distributions.Sampleable}, n::Int, nprime::Int) = if n == nprime == 1 
            ScalarDiffusionStateModel(f, g, init)
        else
            new(f, g, init, n, nprime)
        end
end
    
function DiffusionStateModel(f::Function, g::Function, init::Union{Float64, Array{Float64,1}, Distributions.Sampleable}, n::Int, nprime::Int)
    if n == nprime == 1
        ScalarDiffusionStateModel(f, g, init)
    else
        VectorDiffusionStateModel(f, g, init, n, nprime)
    end
end
        
function DiffusionStateModel(f::Function, g::Function, init::Union{Float64, Array{Float64,1}, Distributions.Sampleable})
    ScalarDiffusionStateModel(f, g, init)
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