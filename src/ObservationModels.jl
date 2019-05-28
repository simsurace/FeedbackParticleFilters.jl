"""
    ObservationModel

Abstract type for the observation model of the filtering problem.

Concrete types: DiffusionObservationModel, PointprocessObservationModel
"""
abstract type ObservationModel end

"""
    DiffusionObservationModel(observation_function::Function, m::Int) <: ObservationModel

A diffusion process observation model dY_t = h(X_t)dt + dV_t, where h is the `observation_function`, X_t is the hidden state at time t,
and V_t is an m-dimensional Brownian motion process.
"""
struct DiffusionObservationModel <: ObservationModel
    observation_function::Function
    m::Int
end

"""
    DiffusionObservationModel1d(observation_function::Function) <: ObservationModel

A diffusion process observation model dY_t = h(X_t)dt + dV_t, where h is the `observation_function`, X_t is the hidden state at time t,
and V_t is a 1-dimensional Brownian motion process.
"""
struct ScalarDiffusionObservationModel <: ObservationModel
    observation_function::Function
end

"""
    PointprocessObservationModel(observation_function::Function, log_observation_function::Function, m::Int) <: ObservationModel

A conditional m-variate Poisson process observation model with intensity h(X_t), where h is the `observation_function' and X_t is the hidden state at time t. The log observation function is represented explicitly for reasons of efficiency.
"""
struct PointprocessObservationModel <: ObservationModel
    observation_function::Function
    log_observation_function::Function
    m::Int
end

struct ScalarPointprocessObservationModel <: ObservationModel
    observation_function::Function
    log_observation_function::Function
end