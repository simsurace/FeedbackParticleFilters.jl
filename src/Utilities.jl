"""
    MSE(trace::AbstractArray)

Compute the MSE of the trace (elements of `trace` have to contain the fields `:state` and `:ensemble`, where the latter is of type `FPFEnsemble`).
"""
function MSE(trace::AbstractArray)
    fnames = fieldnames(eltype(trace))
    if :state in fnames && :ensemble in fnames
        mse = 0.
        for element in trace
            mse += (element.state - StatsBase.mean(element.ensemble.positions))^2
        end
        mse /= length(trace)
    else
        error("Elements of argument miss state and ensemble fields.")
    end#if
end;
    
"""
    RelativeMSE(trace::AbstractArray)

Compute the relative MSE of the trace, i.e. MSE divided by the prior variance (elements of `trace` have to contain the fields `:state` and `:ensemble`, where the latter is of type `FPFEnsemble`).
"""
function RelativeMSE(trace::AbstractArray)
    fnames = fieldnames(eltype(trace))
    if :state in fnames && :ensemble in fnames
        mse = 0.
        var = 0.
        avg = StatsBase.mean([element.state for element in trace])
        for element in trace
            mse += (element.state - StatsBase.mean(element.ensemble.positions))^2
            var += (element.state - avg)^2
        end
        mse /= length(trace)
        var /= length(trace)
        return mse/var
    else
        error("Elements of argument miss state and ensemble fields.")
    end#if
end;