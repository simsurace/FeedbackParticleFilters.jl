"""
    ParametricRepresentation{S, P} <: AbstractFilterRepresentation{S}

Abstract type for the representation of the conditional distribution over the hidden state by a parameter vector containing elements of type `P`.
"""
abstract type ParametricRepresentation{S, P} <: AbstractFilterRepresentation{S} end




"""
    parameter_vector(rep::ParametricRepresentation)

Return a parameter vector describing the representation `rep`.
In the language of differential (information) geometry, the function `parameter_vector` provides a chart on a parametric family of distributions.
"""
function parameter_vector(rep::ParametricRepresentation) end