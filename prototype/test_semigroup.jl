push!(LOAD_PATH,"./")
using PFlib
using Pkg
Pkg.add("StatsBase"); using StatsBase
Pkg.add("LinearAlgebra"); using LinearAlgebra
Pkg.add("Distributed"); using Distributed
Pkg.add("PyPlot"); using PyPlot

# Settings
function p(x::Float64)
    ( exp(-(x-1)^2/(2*0.16))+exp(-(x+1)^2/(2*0.16)) ) / sqrt(8*pi*0.16)
end;

function h(x::Float64)
    x
end;

# numerical integral over R by transform
function finv(y::Float64)
    return tan(pi*y/2)#log((1+y)/(1-y))
end

function finvprime(y::Float64)
    return pi/(1+cos(pi*y))#2/(1-y^2)
end

function NumIntR(fun::Function, n::Int64)
    x = zeros(Float64, n+1)
    int = zeros(Float64, n+1)
    dy = 2. / (n+1)
    y = -1. -dy/2
    for i = 2:(n+1)
        y = y + dy
        x[i] = finv(y)
        int[i] = int[i-1] + fun(x[i]) * dy * finvprime(y)
    end
    x[2:n+1], int[2:n+1]
end;

xx,int = NumIntR(x -> -x*p(x),1000);
K_exact = int ./ p.(xx);

N=1000
testens=UnweightedParticleEnsemble(vcat(0.4.*randn(div(N,2)).-1,0.4.*randn(div(N,2)).+1),N,randn(N),randn(N));

@time Gain_semigroup!(testens, x->x);
@time Gain_semigroup!(testens);

# simulate sequential gain estimation and update
N=1000
testens=UnweightedParticleEnsemble(vcat(0.4.*randn(div(N,2)).-1,0.4.*randn(div(N,2)).+1),N,randn(N),randn(N));
n_time = 1000 # number of time steps
positions = zeros(n_time,N)
potentials = zeros(n_time,N)
gains = zeros(n_time,N)
@time for i = 1:n_time
    Gain_semigroup!(testens)
    positions[i,:] = testens.positions
    potentials[i,:] = testens.potential
    gains[i,:] = testens.gain
    ApplyGain!(testens, 1E-2)
end

plot(positions[:,:]);
