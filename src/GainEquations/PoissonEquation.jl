@doc raw"""
    PoissonEquation(h, ensemble) ::GainEquation

Returns a Poisson equation struct  representing the equation ``\nabla\cdot(p\nabla \phi) = - \tilde h``, where ``p`` is a probability density and ``\tilde h = h-\int h p dx``. 
The container contains the following fields
* `:h': ``h`` itself
* `:positions': an i.i.d. sample from ``p``, represented as a matrix
* `:H': the evaluation of ``h`` at the sample points
* `:mean_H': the sample average of `H'
* `:potential': the evaluation of ``\phi`` at the sample points
* `:gain': the evaluation of ``K=\nabla \phi`` at the sample points

    solve!(eq::PoissonEquation, method::GainEstimationMethod)

Fills the field `:gain' with appropriate values.
The fields `:H', `:mean_H', and `:potential' are stored to be re-used.

    update!(eq::PoissonEquation, ensemble)

Fills the fields `:positions', `:H', and `:mean_H' according to the new samples from `ensemble'.

    update!(eq::PoissonEquation)

Updates fields `:H', and `:mean_H' to be consistent with `:positions'.
"""
struct PoissonEquation{T, TF} <: GainEquation
    h::TF
    positions::Matrix{T}
    H::Matrix{T}
    mean_H::Matrix{T}
    potential::Matrix{T}
    gain::Array{T,3}
    
    PoissonEquation(h, positions, H, mean_H, potential, gain) = 
        if  size(positions, 2) == size(H, 2) == size(potential, 2) == size(gain, 2) && 
            size(H, 1) == size(mean_H, 1) == size(potential, 1) == size(gain, 3) &&
            size(positions, 1) == size(gain, 1) &&
            size(mean_H,2) == 1
            new{eltype(positions), typeof(h)}(h, positions, H, mean_H, potential, gain)
        else
            throw(DimensionMismatch("Inconsistent dimensions of positions, H, mean_H, potential, or gain."))
        end
end
    
function PoissonEquation(h::Function, pos::AbstractMatrix)
    H = mapslices(h, pos, dims=1)
    mean_H = Statistics.mean(H, dims=2)
    return PoissonEquation(h, pos, H, mean_H, zeros(eltype(pos), size(H, 1), size(pos, 2)), zeros(eltype(pos), size(pos, 1), size(pos, 2), size(H, 1)))
end
    

function PoissonEquation(h::Function, ens::UnweightedParticleEnsemble)
    pos = copy(ens.positions)
    return PoissonEquation(h, pos)
end
    
state_dim(eq::PoissonEquation)       = size(eq.positions, 1)
obs_dim(eq::PoissonEquation)         = size(eq.H, 1)
no_of_particles(eq::PoissonEquation) = size(eq.positions, 2)
    
Base.show(io::IO, eq::PoissonEquation) = print(io, "Poisson equation for the gain
    # of particles:        ", size(eq.positions, 2),"
    hidden dimension:      ", state_dim(eq),"
    observed dimension:    ", obs_dim(eq))
    
function Htilde(eq::PoissonEquation)
    return eq.H .- eq.mean_H
end
    


    
    
function update!(eq::PoissonEquation)
    eq.H         .= mapslices(eq.h, eq.positions, dims=1)
    eq.mean_H    .= Statistics.mean(eq.H, dims=2)
    return eq
end
    
function update!(eq::PoissonEquation, pos::AbstractMatrix)
    eq.positions .= pos
    update!(eq)
end    
    
    
function update!(eq::PoissonEquation, ens::UnweightedParticleEnsemble)
    update!(eq, ens.positions)
end
    
    

    
    
    
    
    
function GainEquation(state_model::DiffusionStateModel, obs_model::DiffusionObservationModel, ens::UnweightedParticleEnsemble)
    return PoissonEquation(obs_model.observation_function, ens)
end
    
function GainEquation(state_model::DiffusionStateModel, obs_model::DiffusionObservationModel, N::Int)
    ens = UnweightedParticleEnsemble(state_model, N)
    return GainEquation(state_model, obs_model, ens)
end
    
function GainEquation(filt_prob::AbstractFilteringProblem, ens::UnweightedParticleEnsemble)
    return GainEquation(state_model(filt_prob), obs_model(filt_prob), ens)
end
    
function GainEquation(filt_prob::AbstractFilteringProblem, N::Int)
    return GainEquation(state_model(filt_prob), obs_model(filt_prob), N)
end