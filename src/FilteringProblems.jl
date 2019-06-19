"""
    FilteringProblem{S, T, M1, M2} <: AbstractFilteringProblem{S, T} where {M1<:HiddenStateModel{S}, M2<:ObservationModel{S, T}}

Filtering problem for observations of type `T` and hidden states of type `S`.
"""
struct FilteringProblem{S, T, M1, M2} <: AbstractFilteringProblem{S, T}
    state_model::M1
    obs_model::M2
    FilteringProblem(mod1::HiddenStateModel{S}, mod2::ObservationModel{S, T}) where {S, T} = new{S, T, typeof(mod1), typeof(mod2)}(mod1, mod2)
end

const ContinuousTimeFilteringProblem{S, T, M1<:ContinuousTimeHiddenStateModel{S}, M2<:ContinuousTimeObservationModel{S, T}} = FilteringProblem{S, T, M1, M2};