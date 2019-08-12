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