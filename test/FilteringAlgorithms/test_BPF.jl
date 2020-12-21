using FeedbackParticleFilters, Random, Statistics, LinearAlgebra, Distributions, StatsBase

println("Testing BPF.jl:")

@testset "BPF.jl" begin
    
    print("  inner constructor for BPFState")
    ens   = WeightedParticleEnsemble(randn(3,10))
    state = BPFState(ens)
    print(".")
    @test state.ensemble == ens
    println("DONE.")
    
    print("  inner constructor for BPF")
    f(x)   = -x
    g(x)   = hcat([1.; -2.; 1.])  
    h(x)   = [x[1] * x[2], x[2] * x[3]]
    init   = [0., 0., 0.]
    st_mod = DiffusionStateModel(f, g, init)
    ob_mod = DiffusionObservationModel{Float64, Float64, typeof(h)}(3, 2, h)
    N      = 10
    alpha  = 0.5
    filter = BPF(st_mod, ob_mod, N, alpha)
    print(".")
    @test filter.state_model == st_mod
    print(".")
    @test filter.obs_model == ob_mod
    print(".")
    @test filter.N == N
    print(".")
    @test filter.alpha == alpha
    
    print("  method initial_condition")
    print(".")
    @test initial_condition(filter) ≈ [0., 0., 0.]
    println("DONE.")
    
    print("  method no_of_particles")
    print(".")
    @test no_of_particles(filter) == N
    print(".")
    @test no_of_particles(state) == N
    println("DONE.")
    
    print("  method state_model")
    print(".")
    @test state_model(filter) == st_mod
    println("DONE.")
    
    print("  method obs_model")
    print(".")
    @test obs_model(filter) == ob_mod
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
    
    print("  outer constructors for BPFState")
    state2 = BPFState(filter)
    print(".")
    @test no_of_particles(state2) == N
    print(".")
    @test state2.ensemble.positions == zeros(3,N)
    filt_prob = FilteringProblem(st_mod, ob_mod)
    state3 = BPFState(filt_prob, 2*N)
    print(".")
    @test no_of_particles(state3) == 2*N
    println("DONE.")
    
    print("  outer constructors for BPF")
    filter2 = BPF(filt_prob, N, alpha)
    print(".")
    @test no_of_particles(filter2) == N
    print(".")
    @test filter2.alpha == alpha
    println("DONE.")
    
    print("  method initialize")
    state4 = initialize(filter)
    print(".")
    @test state4 isa BPFState
    print(".")
    @test state4.ensemble.positions == zeros(3,10)
    print(".")
    @test state4.ensemble.weights == StatsBase.ProbabilityWeights(fill(1/N, N))
    println("DONE.")
    
    print("  method propagate!")
    f(x)     = -x
    g(x)     = hcat([1.; -2.; 1.])  
    h(x)     = [x[1] * x[2], x[2] * x[3]]
    init     = [0., 0., 0.]
    st_mod   = DiffusionStateModel(f, g, init)
    ob_mod   = DiffusionObservationModel{Float64, Float64, typeof(h)}(3, 2, h)
    N        = 2
    alpha    = 0.5
    filter   = BPF(st_mod, ob_mod, N, alpha)
    state    = initialize(filter)  
    Random.seed!(0)
    propagate!(state, filter, 0.01)
    print(".")
    @test state.ensemble.positions ≈ [0.06791074260357777 -0.013485387193052173; -0.1656826965800072 -0.11732341492662196; -0.03530074003005963 0.029733585084941616]
    println("DONE.")
    
    print("  method assimilate!")
    ens      = WeightedParticleEnsemble(randn(3,10))
    state    = BPFState(ens)
    oldstate = deepcopy(state)
    f(x)     = -x
    g(x)     = hcat([1.; -2.; 1.])  
    h(x)     = [x[1] * x[2], x[2] * x[3]]
    init     = [0., 0., 0.]
    st_mod   = DiffusionStateModel(f, g, init)
    ob_mod   = DiffusionObservationModel{Float64, Float64, typeof(h)}(3, 2, h)
    N        = 10
    alpha    = 0.5
    filter   = BPF(st_mod, ob_mod, N, alpha)
    dY       = [0.1, 0.2]
    dt       = 0.01
    assimilate!(dY, state, filter, dt)
    print(".")
    @test state.ensemble.positions == oldstate.ensemble.positions
    print(".")
    @test state.ensemble.weights != oldstate.ensemble.weights
    println("DONE.")
    
    print("  method update!")
    init   = [0., 0., 0.]
    st_mod = DiffusionStateModel(f, g, init)
    N      = 10
    alpha  = 0.5
    filter = BPF(st_mod, ob_mod, N, alpha)
    state  = initialize(filter) 
    state2 = initialize(filter)
    Random.seed!(0)
    update!(state, filter, [0.01, -0.02], 0.01)
    Random.seed!(0)
    propagate!(state2, filter, 0.01)
    assimilate!([0.01, -0.02], state2, filter, 0.01)
    print(".")
    @test state.ensemble.positions ≈ state2.ensemble.positions
    print(".")
    @test state.ensemble.weights ≈ state2.ensemble.weights
    println("DONE.")
    
end; #BPF.jl