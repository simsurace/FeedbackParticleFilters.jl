using FeedbackParticleFilters, Random, Statistics, LinearAlgebra

println("Testing FPF.jl:")

@testset "FPF.jl" begin

    print("  inner constructor for FPFState")
    ens   = UnweightedParticleEnsemble(randn(3,10))
    h(x)  = [x[1] * x[2], x[2] * x[3]]
    eq    = PoissonEquation(h, ens)
    state = FPFState(ens, eq)
    print(".")
    @test state.ensemble == ens
    print(".")
    @test state.eq == eq
    println("DONE.")
    
    print("  inner constructor for FPF")
    f(x)   = -x
    g(x)   = [1., -2., 1.]
    init   = [0., 0., 0.]
    st_mod = DiffusionStateModel(f, g, init)
    ob_mod = DiffusionObservationModel{Float64, Float64, typeof(h)}(3, 2, h)
    struct DummyMethod <: GainEstimationMethod end
    method = DummyMethod()
    fpf    = FPF(st_mod, ob_mod, method, state)
    print(".")
    @test fpf.state_model == st_mod
    print(".")
    @test fpf.obs_model == ob_mod
    print(".")
    @test fpf.gain_method == method
    print(".")
    @test fpf.init == state
    
    struct DummyModel <: ObservationModel{Vector{Float64}, Vector{Float64}, ContinuousTime} end
    ob_mod2 = DummyModel()
    print(".")
    @test_throws ErrorException FPF(st_mod, ob_mod2, method, state) # wrong type of observation model
    println("DONE.")
    
end; #FPF.jl