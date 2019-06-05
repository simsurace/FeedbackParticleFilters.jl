#######################
### Gain estimation ###
#######################

"""
    GainEstimationMethod

Any type of gain estimation method for the feedback particle filter. 
Implemented abstract methods (consult individual documentation for concrete methods):

SemigroupMethod
"""
abstract type GainEstimationMethod end


"""
    Solve!(eq::GainEquation, method::GainEstimationMethod)

Solve the gain equation `eq` using method `method`.
"""
function Solve!(eq::GainEquation, method::GainEstimationMethod) end

"""
    Update!(eq::GainEquation, ensemble::FPFEnsemble)

Update the gain equation `eq` using data from particle ensemble `ensemble`, i.e. compute observation function at particle positions.
"""
function Update!(eq::GainEquation, ensemble::FPFEnsemble) end

function Update!(eq::ScalarPoissonEquation, ensemble::FPFEnsemble)
    eq.positions = ensemble.positions
    broadcast!(eq.h, eq.H, eq.positions)
    eq.mean_H = StatsBase.mean(eq.H)
end

function Update!(eq::VectorScalarPoissonEquation, ensemble::FPFEnsemble)
    eq.positions = ensemble.positions
    broadcast!(eq.h, eq.H, eq.positions)
    eq.mean_H = StatsBase.mean(eq.H)
end

"""
    FPFUpdater(filt_prob::ContinuousTimeFilteringProblem, method::GainEstimationMethod)

Returns a function called `update!` that assimilates one observation by solving the gain estimation problem and then updating the particles.
"""
function FPFUpdater(filt_prob::ContinuousTimeFilteringProblem, method::GainEstimationMethod, dt::Float64)
    function update!(ensemble::FPFEnsemble, eq::ScalarPoissonEquation, obs)
        Update!(eq, ensemble)
        Solve!(eq, method)
        error = obs .- eq.mean_H * dt / 2 .- eq.H .* dt ./ 2
        ApplyGain!(ensemble, eq, error)
    end
end
        
