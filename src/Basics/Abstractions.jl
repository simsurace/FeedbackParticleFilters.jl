"""
    TimeType

Abstract type for time.
"""
abstract type TimeType end




"""
    ContinuousTime <: TimeType

Abstract type for processes that run in continuous time and need to be discretized.
"""
abstract type ContinuousTime <: TimeType end




"""
    DiscreteTime <: TimeType

Abstract type for processes that run in discrete time without a notion of physical units (i.e. time steps).
"""
abstract type DiscreteTime <: TimeType end




