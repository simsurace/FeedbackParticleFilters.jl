abstract type Simulation end









struct ContinuousTimeSimulation{FP<:AbstractFilteringProblem, FA<:AbstractFilteringAlgorithm, DT<:Real} <: Simulation
    filt_prob::FP
    filt_algo::FA
    no_of_timesteps::Int
    dt::DT
end




function Base.show(io::IO, sim::ContinuousTimeSimulation)
    print(io, "Continuous-time simulation
    filtering problem:                      ", sim.filt_prob,"
    filtering algorithm:                    ", sim.filt_algo,"
    number of time steps:                   ", sim.no_of_timesteps,"
    size of time step:                      ", sim.dt)
end



"""
    run!(simulation)

Runs the simulation.
"""
function run!(simulation::Simulation; records = ())
    sfs = SimulationState(simulation.filt_prob, simulation.filt_algo)
    out = Array{Any, 2}(undef, length(records), simulation.no_of_timesteps)
    
    for t in 1:simulation.no_of_timesteps
        propagate!(sfs, simulation.filt_prob, simulation.filt_algo, simulation.dt)
        for (i,record) in enumerate(records)
            out[i,t] = deepcopy(record(sfs))
        end
    end
    return out
end


"""
    simulate!(filtering_algorithm, filtering_problem, no_of_timesteps, dt)

Runs a simulation of the hidden state, observation, and filtering algorithm for a duration of `no_of_timesteps`.
"""
function simulate!(filt_algo::AbstractFilteringAlgorithm{ContinuousTime, ContinuousTime}, filt_prob::AbstractFilteringProblem{S1, S2, ContinuousTime, ContinuousTime}, n_time, dt) where {S1, S2}
    simulation = ContinuousTimeSimulation(filt_prob, filt_algo, n_time, dt)
    run!(simulation)
end