"""
    FilteringProblem(state_model, obs_model)

Specify a filtering problem for a hidden state model `state_model` and observation model `obs_model`.

Constraints: `state_model` and `obs_model` must be chosen such that
* `state_model` is of type `HiddenStateModel{S1, T1}`,
* `obs_model` is of type `ObservationModel{S1, S2, T2}`,
where `T1<:TimeType`, `T2 <:TimeType`, and `S1`, `S2` are arbitrary.
This ensures that the filtering problem can be evaluated, i.e. the hidden states are of appropriate type to generate observations.
"""
struct FilteringProblem{S1, S2, T1, T2, M1, M2} <: AbstractFilteringProblem{S1, S2, T1, T2}
    state_model::M1
    obs_model::M2
    function FilteringProblem(mod1::HiddenStateModel{S1, T1}, mod2::ObservationModel{S1, S2, T2}) where {S1, S2, T1<:TimeType, T2<:TimeType} 
        return new{S1, S2, T1, T2, typeof(mod1), typeof(mod2)}(mod1, mod2)
    end
end


# mandatory methods
state_model(problem::FilteringProblem)      = problem.state_model
obs_model(problem::FilteringProblem)        = problem.obs_model







# optional methods
function Base.show(io::IO, problem::FilteringProblem{S1, S2, T1, T2, M1, M2}) where {S1, S2, T1, T2, M1, M2}
    println(io, "Filtering problem with")
    println(io, "    ", T1, " hidden state of type     ", S1)
    println(io, "    ", T2, " observation of type      ", S2)
    println(io, "Call state_model and obs_model in order to display further information.")
end