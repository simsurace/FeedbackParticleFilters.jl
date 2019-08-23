module FeedbackParticleFilters

using LinearAlgebra
using Distributions
using Statistics
using Random
using PDMats

import Statistics.mean
import Statistics.cov
import Statistics.var

####################################
########## CORE FUNCTIONS ##########
####################################

include("Basics/Abstractions.jl")
export
    TimeType,
    DiscreteTime,
    ContinuousTime

include("Basics/AbstractModel.jl")
export
    AbstractModel,
    state_type,
    obs_type,
    time_type

include("Basics/AbstractFilteringProblem.jl")
export
    AbstractFilteringProblem,
    state_model,
    obs_model,
    state_dim,
    obs_dim,
    hidden_time_type,
    obs_time_type
   
include("Basics/AbstractFilterRepresentation.jl")
export
    AbstractFilterRepresentation,
    represented_type,
    dim

include("Basics/AbstractFilteringAlgorithm.jl")
export
    AbstractFilteringAlgorithm,
    AbstractFilterState,
    update!

include("Basics/HiddenStateModel.jl")
export
    HiddenStateModel,
    state_dim,
    initial_condition,
    state_type,
    time_type

include("Basics/ObservationModel.jl")
export
    ObservationModel,
    state_dim,
    obs_dim,
    state_type,
    obs_type,
    time_type,
    emit!

include("Basics/ParametricRepresentation.jl")
export
    ParametricRepresentation,
    parameter_vector

include("Basics/ParticleRepresentation.jl")
export
    ParticleRepresentation,
    get_pos,
    list_of_pos,
    no_of_particles,
    eff_no_of_particles,
    particle_dim,
    dim,
    propagate!

include("Basics/UnweightedParticleRepresentation.jl")
export
    UnweightedParticleRepresentation

include("Basics/WeightedParticleRepresentation.jl")
export
    WeightedParticleRepresentation,
    get_weight,
    list_of_weights,
    eff_no_of_particles,
    dim

include("Basics/GainEstimation.jl")
export
    GainEquation,
    GainEstimationMethod,
    solve!,
    update!,
    state_dim,
    obs_dim

include("Basics/FilteringProblem.jl")
export
    FilteringProblem



####################################
########### STATE MODELS ###########
####################################

include("StateModels/DiffusionStateModel.jl")
export
    DiffusionStateModel,
    ScalarDiffusionStateModel,
    drift,
    diffusion,
    initial_condition,
    state_dim,
    noise_dim,
    initialize,
    drift_function,
    diffusion_function

include("StateModels/LinearDiffusionStateModel.jl")
export
    LinearDiffusionStateModel,
    find_stationary_variance,
    drift,
    diffusion,
    initial_condition,
    initialize,
    state_dim,
    noise_dim



####################################
######## OBSERVATION MODELS ########
####################################

include("ObservationModels/DiffusionObservationModel.jl")
export
    DiffusionObservationModel,
    ScalarDiffusionObservationModel,
    initial_condition,
    drift,
    state_dim,
    obs_dim,
    noise_dim,
    observation_function

include("ObservationModels/LinearDiffusionObservationModel.jl")
export
    LinearDiffusionObservationModel,
    find_stationary_variance,
    drift,
    state_dim,
    obs_dim,
    noise_dim,
    observation_function



####################################
###### FILTER REPRESENTATIONS ######
####################################

include("FilterRepresentations/MultivariateGaussian.jl")
export
    MultivariateGaussian,
    parameter_vector

include("FilterRepresentations/UnweightedParticleEnsemble.jl")
export
    UnweightedParticleEnsemble



####################################
########## GAIN EQUATIONS ##########
####################################

include("GainEquations/PoissonEquation.jl")
export
    PoissonEquation,
    Htilde



####################################
##### GAIN ESTIMATION METHODS ######
####################################

include("GainEstimationMethods/ConstantGainApproximation.jl")
export
    ConstantGainApproximation

include("GainEstimationMethods/SemigroupMethod.jl")
export
    SemigroupMethod

include("GainEstimationMethods/DifferentialRKHSMethod.jl")
export
    DifferentialRKHSMethod



####################################
####### FILTERING ALGORITHMS #######
####################################

# Kalman-Bucy filter
include("FilteringAlgorithms/KBF.jl")
export
    KBF,
    KBState,
    initialize

# Feedback Particle Filter (original)
include("FilteringAlgorithms/FPF.jl")
export
    FPF,
    FPFState,
    gain_estimation_method,
    propagate!, 
    assimilate!,
    heun!,
    gainxerror,
    add_gainxerror!,
    applygain!



####################################
############ SIMULATION ############
####################################

include("Simulation/Simulation.jl")
export
    Simulation,
    ContinuousTimeSimulation,
    run!,
    simulate!

# Feedback Particle Filter (original)
include("Simulation/SimulationState.jl")
export
    SimulationState,
    hidden_state,
    obs,
    filter_state,
    cond_mean,
    cond_cov,
    cond_var,
    propagate!



end # module