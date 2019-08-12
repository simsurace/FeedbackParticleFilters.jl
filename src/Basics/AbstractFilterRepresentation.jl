"""
    AbstractFilterRepresentation{S}

Abstract type for representation of the conditional distribution over the hidden state of type `S`.
"""
abstract type AbstractFilterRepresentation{S} end




"""
    represented_type(rep::AbstractFilterRepresentation)

Return the type which is represented by `rep`.
"""
represented_type(rep::AbstractFilterRepresentation{S}) where S = S




"""
    dim(rep::AbstractFilterRepresentation{S})

Return the dimensionality of the filter representation `rep`.
"""
function dim(rep::AbstractFilterRepresentation{S}) where S end




# How to add new filter representations:
# * Add a struct which is a subtype of AbstractFilterRepresentation{S}, where S is the type to be represented
# * Implement a method for dim which returns the dimensionality of the representation.




