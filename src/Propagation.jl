########################
### Gain application ###
########################


"""
    ApplyGain!(ensemble::FPFEnsemble, eq::GainEquation, dt::Float64)

Update the particle positions `ensemble.positions` by applying the gain stored in `eq.gain`.
"""
function ApplyGain!(ensemble::FPFEnsemble, eq::GainEquation, dt::Float64)
   broadcast!(+, ensemble.positions, ensemble.positions, dt .* eq.gain)
end;

function ApplyGain!(ensemble::FPFEnsemble, eq::GainEquation, error::Array{Float64,1})
   broadcast!(+, ensemble.positions, ensemble.positions, eq.gain .* error)
end;