abstract type UnweightedParticleEnsemble{T} end;
abstract type WeightedParticleEnsemble{T} end; #currently not being used
abstract type ObservationData{T2} end;
abstract type GainData{T1,T2} end;

########################################
### Concrete particle ensemble types ###
########################################


"""
    FPFEnsemble{T1,T2}(positions::Array{T1,1}, size::Int64, obs_data::ObservationData{T2}, gain_data::GainData{T1,T2})

Feedback particle filter ensemble state that keeps track of particle `positions` of type `T1`, observation data (whose structure varies depending on the observation model), and gain data (whose structure varies depending on the gain estimation method).
"""
mutable struct FPFEnsemble{T1,T2} <: UnweightedParticleEnsemble{T1}
    positions::Array{T1,1}
    size::Int64
    obs_data::ObservationData{T2}
    gain_data::GainData{T1,T2}
end
