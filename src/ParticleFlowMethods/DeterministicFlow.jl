"""
    DeterministicFlow(n, method[, projection])

Deterministic log-homotopy flow discretized by `n` steps, where vector field is estimated using `method`.

### Optional arguments
- `projection`: a function to be applied after every step of the flow.
"""
struct DeterministicFlow{TG<:GainEstimationMethod, TP<:Function} <: ParticleFlowMethod
    n_steps::Int
    gain_method::TG
    projection::TP
end

DeterministicFlow(n, method) = DeterministicFlow(n, method, identity)




#####################
### BASIC METHODS ###
#####################
                    
function Base.show(io::IO, ::MIME"text/plain", flow::DeterministicFlow)
    print(io, "Deterministic particle flow
    gain estimation method:                 ", flow.gain_method, "
    number of steps:                        ", flow.n_steps)
end

function Base.show(io::IO, flow::DeterministicFlow)
    print(io, "Deterministic particle flow with ", flow.n_steps, " steps and ", flow.gain_method)
end






######################
### MAIN ALGORITHM ###
######################

function flow!(ensemble::UnweightedParticleEnsemble{T}, log_likelihood::Function, method::DeterministicFlow) where T
    eq = PoissonEquation(log_likelihood, ensemble)
    if eq.H == zeros(T, size(eq.H)) #nothing to compute
        foreach(method.projection, eachcol(ensemble.positions))
        return ensemble.positions
    end
    
    n = method.n_steps
    for i in 1:n
        solve!(eq, method.gain_method)
        ensemble.positions .+= eq.gain[:,:,1]/n
        foreach(method.projection, eachcol(ensemble.positions))
        update!(eq, ensemble)
    end
    return ensemble.positions
end