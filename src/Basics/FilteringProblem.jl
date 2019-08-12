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







# pretty printing
function Base.show(io::IO, ::MIME"text/plain", problem::FilteringProblem{S1, S2, T1, T2, M1, M2}) where {S1, S2, T1, T2, M1, M2}
    print(io, T1, " - ", T2," filtering problem
    hidden state model:                     ", problem.state_model,"
    observation model:                      ", problem.obs_model,"
    hidden state type:                      ", S1,"
    observation type:                       ", S2)
end

function Base.show(io::IO, problem::FilteringProblem{S1, S2, T1, T2, M1, M2}) where {S1, S2, T1, T2, M1, M2}
    print(io, T1, " - ", T2," with ", M1, " and ", M2)
end