"""
    propagate(state::S, state_model::HiddenStateModel{S}) where S

Propagate the hidden `state` by one time step according to `state_model`.
"""
function propagate(state::S, state_model::HiddenStateModel{S}) where S end




#function (state_model::M)(state::S) where M<:HiddenStateModel{S} where S
    #propagate(state, state_model)
#end







"""
    propagate!(state, state_model) -> state

In-place version of `propagate`.
"""
function propagate!(state::S, state_model::HiddenStateModel{S}) where S 
    state = propagate(state, state_model)
end


"""
    propagate(ensemble::ParticleRepresentation{S}, state_model::HiddenStateModel{S}) where S

Propagate each member of the particle ensemble by one time step according to `state_model`.
"""
function propagate(ensemble::ParticleRepresentation{S}, state_model::HiddenStateModel{S}) where S
    Map(propagate, ensemble)
end


"""
    propagate(filter_rep::AbstractFilterRepresentation{S}, state_model::HiddenStateModel{S}) where S

Propagate each member of the particle ensemble by one time step according to `state_model`.
"""
function propagate(filter_rep::AbstractFilterRepresentation{S}, state_model::HiddenStateModel{S}) where S end






"""
    eltype(filter_rep::T) where T<:AbstractFilterRepresentation{S} where S <: AbstractHiddenState

Extract S.
"""
function Base.eltype(filter_rep::T) where T<:AbstractFilterRepresentation{S} where S
    S
end


"""
    Map(f::T, A::AbstractArray) where T<:Function

Same as `Base.map`.
"""
function Map(f::T, A::AbstractArray) where T<:Function 
    map(f, A)
end


"""
    Map(F::NTuple{N, Function}, x::T) where {N, T<:Number}

Output an array with the results of applying each f in F to x.
"""
function Map(F::NTuple{N, Function}, x::T) where {N, T<:Number} 
    [f(x) for f in F]
end;


"""
    Map(F::AbstractArray{Function}, x::T) where {N, T<:Number} 

Output an array with the results of applying each f in F to x.
"""
function Map(F::AbstractArray{Function}, x::T) where {N, T<:Number} 
    [f(x) for f in F]
end;


"""
    Map(F::AbstractArray{U}, x::T) where {N, T<:Number, U<:Function} 

Output an array with the results of applying each f in F to x.
"""
function Map(F::AbstractArray{U}, x::T) where {N, T<:Number, U<:Function}
    [f(x) for f in F]
end;


"""
    Map(F::NTuple{N, Function}, A::AbstractArray; output_shape=2) where N

Output an Array of applying each f in F to each a in A, where `output_shape` determines whether the output should inherit the shape of F or A.
"""
function Map(F::NTuple{N, Function}, A::AbstractArray; output_shape=2) where N
    if output_shape == 1 
        try
            collect(Map.(F, Ref(A)))
        catch
            error("ERROR: functions cannot be applied at first level. Call with output_shape=2.")
        end
    elseif output_shape == 2
        try
            [f(A) for f in F]
        catch
            collect(Map.(Ref(F), A))
        end
    else
        error("ERROR: Invalid output_shape parameter. Must be either 1 or 2.") 
    end
end;
                

                

                
                
                
                
"""
    Map(F::AbstractArray{T}, A::AbstractArray; output_shape=2) where {N, T<:Function}

Output an Array of applying each f in F to each a in A, where `output_shape` determines whether the output should inherit the shape of F or A.
I.e. if `output_shape=1`, then for each f in F the output contains an element which contains f applied to all elements from A.
If `output_shape=2`, then for each a in A the output contains an element which contains all f's in F applied to a.
`Map` also implements recursion when `output_shape=2`: `Map` starts at the first level and recursively tries to map to each level until a method has been found.
If `output_shape=1`, and error is thrown if the functions in F cannot be applied at the first level of A.
"""
function Map(F::AbstractArray{T}, A::AbstractArray; output_shape=2) where {N, T<:Function}
    if output_shape == 1 
        try
            collect(Map.(F, Ref(A)))
        catch
            error("ERROR: functions cannot be applied at first level. Call with output_shape=2.")
        end
    elseif output_shape == 2
        try
            [f(A) for f in F]
        catch
            collect(Map.(Ref(F), A))
        end
    else
        error("ERROR: Invalid output_shape parameter. Must be either 1 or 2.") 
    end
end;

 
                                
                                
"""
    Map(f::U, ensemble::T) where {U<:Function, T<:UnweightedParticleRepresentation{S}} where S<:AbstractHiddenState

Short-hand for `Base.map` applied to `ensemble.positions`.
"""
function Map(f::U, ensemble::T) where {U<:Function, T<:UnweightedParticleRepresentation{S}} where S<:AbstractHiddenState
    if !hasmethod(f, tuple(S))
        error("ERROR: this function cannot be evaluated for this ensemble of particles.")
    end
    Map(f, ensemble.positions)
end;
                                

                                
                                
                                
                                
                                
                                
                                
"""
    Map(F::NTuple{N, U}, ensemble::T) where {N, U<:Function, T<:UnweightedParticleRepresentation{S}} where S<:AbstractHiddenState

Short-hand for `Map` applied to `ensemble.positions`.
"""
function Map(F::NTuple{N, U}, ensemble::T) where {N, U<:Function, T<:UnweightedParticleRepresentation{S}} where S<:AbstractHiddenState
    if !hasmethod(f, tuple(S))
        error("ERROR: this function cannot be evaluated for this ensemble of particles.")
    end
    Map(F, ensemble.positions)
end;