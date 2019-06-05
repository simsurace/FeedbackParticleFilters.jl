abstract type UnweightedParticleEnsemble{T} end;
abstract type WeightedParticleEnsemble{T} end; #currently not being used

########################################
### Concrete particle ensemble types ###
########################################


"""
    FPFEnsemble{T}

Feedback particle filter ensemble state that keeps track of particle `positions` of type `T` and the ensemble size.
"""
mutable struct FPFEnsemble{T} <: UnweightedParticleEnsemble{T}
    positions::Array{T,1}
    size::Int64
    FPFEnsemble(positions::Array, size::Int) = if length(positions) != size
                                        if length(positions) == 1
                                            new{typeof(positions[1])}(fill(positions[1], size), size)
                                        else
                                            error("ERROR: number of particle positions does not equal ensemble size")
                                        end
                                   else
                                        new{typeof(positions[1])}(positions, size)
                                   end
end
        
FPFEnsemble(distribution::Distributions.Sampleable, N::Int) = FPFEnsemble(rand(distribution, N), N)

