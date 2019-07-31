struct PoissonEquation{T, TF} <: GainEquation
    h::TF
    positions::Matrix{T}
    H::Matrix{T}
    mean_H::Matrix{T}
    potential::Matrix{T}
    gain::Array{T,3}
    
    PoissonEquation(h, positions, H, mean_H, potential, gain) = 
        if size(positions, 2) == size(H, 2) == size(potential, 2) == size(gain, 2) && size(H, 1) == size(mean_H, 1) == size(potential, 1) == size(gain, 3)
            new{eltype(positions), typeof(h)}(h, positions, H, mean_H, potential, gain)
        else
            error("ERROR: length mismatch")
        end
end
    
function PoissonEquation(h::Function, pos::AbstractMatrix)
    H = mapslices(h, pos, dims=1)
    mean_H = StatsBase.mean(H, dims=2)
    return PoissonEquation(h, pos, H, mean_H, zeros(eltype(pos), size(H, 1), size(pos, 2)), zeros(eltype(pos), size(pos, 1), size(pos, 2), size(H, 1)))
end
    
Base.show(io::IO, eq::PoissonEquation) = print(io, "Poisson equation for the gain
    # of particles:        ", size(eq.positions, 2),"
    hidden dimension:      ", size(eq.positions, 1),"
    observed dimension:    ", size(eq.H, 1))
    
function Htilde(eq::PoissonEquation)
    return eq.H .- eq.mean_H
end
    
    
    
    
    
    
    
    
    
    
function update!(eq::PoissonEquation, ens::UnweightedParticleEnsemble)
    eq.positions .= ens.positions
    eq.H         .= mapslices(eq.h, eq.positions, dims=1)
    eq.mean_H    .= StatsBase.mean(eq.H, dims=2)
end