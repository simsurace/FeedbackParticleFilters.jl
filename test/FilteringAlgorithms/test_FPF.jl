using FeedbackParticleFilters, Random, Statistics, LinearAlgebra, Distributions, FillArrays

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
    g(x)   = hcat([1.; -2.; 1.])  
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
    @test initial_condition(fpf) ≈ [0., 0., 0.]
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
    @test Statistics.mean(state) ≈ mean(ens)
    println("DONE.")
    
    print("  method for Statistics.cov")
    print(".")
    @test Statistics.cov(state) ≈ cov(ens)
    println("DONE.")
    
    print("  method for Statistics.var")
    print(".")
    @test Statistics.var(state) ≈ var(ens)
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
    @test state4.ensemble.positions ≈ zeros(3,10)
    println("DONE.")
    
    print("  method assimilate!")
    function FeedbackParticleFilters.solve!(eq::PoissonEquation, method::DummyMethod)
        eq.gain .= Statistics.mean(eq.positions) * ones(size(eq.gain))
    end
    
    init   = MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1., 3)))
    st_mod = DiffusionStateModel(f, g, init)
    ob_mod = DiffusionObservationModel{Float64, Float64, typeof(h)}(3, 2, h)
    method = DummyMethod()
    fpf    = FPF(st_mod, ob_mod, method, 10)
    state  = initialize(fpf)
    
    oldstate = deepcopy(state)
    solve!(oldstate.eq, method)
    print(".")
    @test oldstate.eq.gain == Statistics.mean(oldstate.eq.positions) * ones(3,10,2) # test DummyMethod
    
    assimilate!([0.01, -0.02], state, fpf, 0.01)
    error = [0.01, -0.02] .- 0.01*oldstate.eq.H/2 .- 0.01*oldstate.eq.mean_H/2
    heun!(oldstate.eq, oldstate.ensemble, error, method)
    print(".")
    @test state.eq.gain ≈ oldstate.eq.gain
    
    print(".")
    @test state.eq.positions ≈ state.ensemble.positions
    println("DONE.")
    
    print("  method propagate!")
    init   = [0., 0., 0.]
    st_mod = DiffusionStateModel(f, g, init)
    fpf    = FPF(st_mod, ob_mod, method, 2)
    state  = initialize(fpf)
    state1 = deepcopy(state)
    propagate!(state1, fpf, 0.01)
    print(".")
    @test all(state.ensemble.positions .!= state1.ensemble.positions)
    println("DONE.")
    
    print("  method update!")
    init   = [0., 0., 0.]
    st_mod = DiffusionStateModel(f, g, init)
    fpf    = FPF(st_mod, ob_mod, method, 2)
    state  = initialize(fpf) 
    state2 = initialize(fpf)
    Random.seed!(0)
    update!(state, fpf, [0.01, -0.02], 0.01)
    Random.seed!(0)
    propagate!(state2, fpf, 0.01)
    assimilate!([0.01, -0.02], state2, fpf, 0.01)
    print(".")
    @test state.ensemble.positions ≈ state2.ensemble.positions
    print(".")
    @test state.eq.gain ≈ state2.eq.gain
    println("DONE.")
    
end; #FPF.jl
