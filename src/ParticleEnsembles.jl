"""
    UnweightedParticleEnsemble{T<:AbstractHiddenState}

Concrete mutable type for the representation of the conditional distribution over the hidden state by unweighted particles (samples) with no additional structure.
"""
mutable struct UnweightedParticleEnsemble{T<:AbstractHiddenState} <: UnweightedParticleRepresentation{T}
    positions::AbstractVector{T}
    size::Int
    
    UnweightedParticleEnsemble(positions::AbstractVector{T}, size::Int) where T<:AbstractHiddenState = 
        if length(positions) != size
            if length(positions) == 1
                new{T}(fill(positions[1], size), size)
            else
                error("ERROR: number of particle positions does not equal ensemble size.")
            end
        else
            new{T}(positions, size)
        end
end;

        
########################################
### Concrete particle ensemble types ###
########################################


"""
    FPFEnsemble{T}

Feedback particle filter ensemble state that keeps track of particle `positions` of type `T` and the ensemble size.
"""
mutable struct FPFEnsemble{T<:AbstractHiddenState} <: UnweightedParticleRepresentation{T}
    positions::AbstractVector{T}
    size::Int
    eq::AbstractGainEquation{T}
    FPFEnsemble(positions::AbstractVector{T}, size::Int) where T<:AbstractHiddenState = 
        if length(positions) != size
            if length(positions) == 1
                new{T}(fill(positions[1], size), size, EmptyGainEquation{T}())
            else
                error("ERROR: number of particle positions does not equal ensemble size.")
            end
        else
            new{T}(positions, size, EmptyGainEquation{T}())
        end
end;
        
FPFEnsemble(distribution::Distributions.Sampleable, N::Int) = FPFEnsemble(rand(distribution, N), N)


