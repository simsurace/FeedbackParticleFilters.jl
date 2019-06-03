abstract type Filter end
abstract type ContinuousTimeFilter <: Filter end
    
abstract type Simulation end
    

mutable struct FeedbackParticleFilter <: ContinuousTimeFilter
    filt_prob::ContinuousTimeFilteringProblem
    method::GainEstimationMethod
    N::Int
    FeedbackParticleFilter(filt_prob::ContinuousTimeFilteringProblem, method::GainEstimationMethod, N::Int) = new(filt_prob, method, N)
end

mutable struct FPFSimulation{T1, T2} <: Simulation
    n_time::Int
    dt::Float64
    propagate!::Function
    emit::Function
    update!::Function
    state::T1
    observation::T2
    ensemble::FPFEnsemble
    eq::GainEquation
end

function FPFSimulation(filter::FeedbackParticleFilter, n_time::Int, dt::Float64)
    filt_prob = filter.filt_prob
    method = filter.method
    N = filter.N
    propagate! = Propagator(filt_prob.state_model, dt)
    emit = Emitter(filt_prob.obs_model, dt)
    update! = FPFUpdater(filt_prob, method, dt)
    state = rand(filt_prob.state_model.initial_distribution)
    observation = emit(state)
    ensemble = FPFEnsemble(filt_prob.state_model, N)
    eq = GainEquation(filt_prob, ensemble)
    FPFSimulation(n_time, dt, propagate!, emit, update!, state, observation, ensemble, eq)
end

function run!(simulation::FPFSimulation)
    print("Starting simulation...")
    for i in 1:simulation.n_time
        simulation.state = simulation.propagate!(simulation.state)
        simulation.observation = simulation.emit(simulation.state)
        simulation.propagate!(simulation.ensemble)
        simulation.update!(simulation.ensemble, simulation.eq, simulation.observation)
    end
    println("DONE.")
end