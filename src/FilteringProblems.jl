abstract type FilteringProblem end

struct ContinuousTimeFilteringProblem <: FilteringProblem
    state_model::StateModel
    obs_model::ObservationModel
end

