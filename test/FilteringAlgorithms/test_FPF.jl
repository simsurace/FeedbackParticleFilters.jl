using FeedbackParticleFilters, Random, Statistics, LinearAlgebra, Distributions

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
    fpf    = FPF(st_mod, ob_mod, method, 10)
    print(".")
    @test fpf.state_model == st_mod
    print(".")
    @test fpf.obs_model == ob_mod
    print(".")
    @test fpf.gain_method == method
    print(".")
    @test fpf.N == 10
    
    struct DummyModel <: ObservationModel{Vector{Float64}, Vector{Float64}, ContinuousTime} end
    ob_mod2 = DummyModel()
    print(".")
    @test_throws ErrorException FPF(st_mod, ob_mod2, method, 10) # wrong type of observation model
    println("DONE.")
    
    print("  method initial_condition")
    print(".")
    @test initial_condition(fpf) == [0., 0., 0.]
    println("DONE.")
    
    print("  method no_of_particles")
    print(".")
    @test no_of_particles(fpf) == 10
    print(".")
    @test no_of_particles(state) == 10
    println("DONE.")
    
    print("  method state_model")
    print(".")
    @test state_model(fpf) == st_mod
    println("DONE.")
    
    print("  method obs_model")
    print(".")
    @test obs_model(fpf) == ob_mod
    println("DONE.")
    
    print("  method gain_estimation_method")
    print(".")
    @test gain_estimation_method(fpf) == method
    println("DONE.")
    
    print("  method for Statistics.mean")
    print(".")
    @test Statistics.mean(state) == mean(ens)
    println("DONE.")
    
    print("  method for Statistics.cov")
    print(".")
    @test Statistics.cov(state) == cov(ens)
    println("DONE.")
    
    print("  method for Statistics.var")
    print(".")
    @test Statistics.var(state) == var(ens)
    println("DONE.")
    
    print("  outer constructors for FPFState")
    state2 = FPFState(fpf)
    print(".")
    @test no_of_particles(state2) == 10
    print(".")
    @test state2.ensemble.positions == zeros(3,10)
    filt_prob = FilteringProblem(st_mod, ob_mod)
    state3 = FPFState(filt_prob, 20)
    print(".")
    @test no_of_particles(state3) == 20
    println("DONE.")
    
    print("  outer constructors for FPF")
    fpf2 = FPF(filt_prob, method, 100)
    print(".")
    @test no_of_particles(fpf2) == 100
    println("DONE.")
    
    print("  method initialize")
    state4 = initialize(fpf)
    print(".")
    @test state4 isa FPFState
    print(".")
    @test state4.ensemble.positions == zeros(3,10)
    println("DONE.")
    
    print("  method assimilate!")
    function FeedbackParticleFilters.solve!(eq::PoissonEquation, method::DummyMethod)
        eq.gain .= ones(size(eq.gain))
    end
    
    init   = MvNormal(3, 1.)
    st_mod = DiffusionStateModel(f, g, init)
    ob_mod = DiffusionObservationModel{Float64, Float64, typeof(h)}(3, 2, h)
    method = DummyMethod()
    fpf    = FPF(st_mod, ob_mod, method, 10)
    state  = initialize(fpf)
    
    oldstate = deepcopy(state)
    solve!(oldstate.eq, method)
    print(".")
    @test oldstate.eq.gain == ones(3,10,2) # test DummyMethod
    
    assimilate!([0.01, -0.02], state, fpf, 0.01)
    print(".")
    @test state.eq.gain == ones(3,10,2)
    
    for i in 1:10
        print(".")
        @test state.ensemble.positions[:,i] ≈ oldstate.ensemble.positions[:,i] + ones(3,2) * ([0.01, -0.02] - 0.01*oldstate.eq.H[:,i]/2 - 0.01*oldstate.eq.mean_H[:,1]/2)
    end
    
    print(".")
    @test state.eq.positions == state.ensemble.positions
    println("DONE.")
    
end; #FPF.jl