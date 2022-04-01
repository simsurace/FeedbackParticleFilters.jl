using FeedbackParticleFilters, Random, Statistics, LinearAlgebra, Distributions, FillArrays

println("Testing ppFPF.jl:")

@testset "ppFPF.jl" begin
    # helper function for keeping particles between 0 and 2*pi
    function mod2pi!(x::AbstractArray)
        for i in eachindex(x)
            x[i] = mod2pi(x[i])
        end
    end

    print("  inner constructor for ppFPFState")
    ens = UnweightedParticleEnsemble(2π .* randn(1,10))
    function h(x)
        20*exp.(10*cos.(x[1] .- 2*pi*collect(0:3)/4))/exp(10.)
    end
    h(x::AbstractMatrix) = mapslices(h, x, dims=1)
    eq_dt  = PoissonEquation(x -> -sum(h(x)), ens)
    eq_dN  = PoissonEquation(x -> log.(h(x)), ens)
    state = ppFPFState(ens, eq_dN, eq_dt)
    print("-")
    @test state.ensemble == ens
    print("-")
    @test state.eq_dt == eq_dt
    print("-")
    @test state.eq_dN == eq_dN
    println("DONE")

    
    print("  inner constructor for ppFPF")
    st_mod = BrownianMotionCircle()
    ob_mod = CountingObservationModel{Float64, Int, typeof(h)}(1, 4, h)
    filt_prob = FilteringProblem(st_mod, ob_mod)
    g_method = DifferentialRKHSMethodS1(0.1, 0.01)
    f_method = DeterministicFlow(100, g_method, mod2pi!)
    fpf = ppFPF(st_mod, ob_mod, g_method, f_method, 10)
    print("-")
    @test fpf.state_model == st_mod
    print("-")
    @test fpf.obs_model == ob_mod
    print("-")
    @test fpf.gain_method == g_method
    print("-")
    @test fpf.flow_method == f_method
    print("-")
    @test fpf.N == 10
    println("DONE")

    print("  method initial_condition")
    print("-")
    @test initial_condition(fpf) == Uniform(0,2π)
    println("DONE")

    print("  method no_of_particles")
    print("-")
    @test no_of_particles(fpf) == 10
    print("-")
    @test no_of_particles(state) == 10
    println("DONE")

    print("  method state_model")
    print("-")
    @test state_model(fpf) == st_mod
    println("DONE")

    print("  method obs_model")
    print("-")
    @test obs_model(fpf) == ob_mod
    println("DONE")

    print("  method gain_estimation_method")
    print("-")
    @test gain_estimation_method(fpf) == g_method
    println("DONE")

    print("  method gain_estimation_method")
    print("-")
    @test flow_method(fpf) == f_method
    println("DONE")
    
    print("  method for Statistics.mean")
    print("-")
    @test Statistics.mean(state) ≈ mean(ens)
    println("DONE")
    
    print("  method for Statistics.cov")
    print("-")
    @test Statistics.cov(state) ≈ cov(ens)
    println("DONE")
    
    print("  method for Statistics.var")
    print("-")
    @test Statistics.var(state) ≈ var(ens)
    println("DONE")

    print("  outer constructors for ppFPFState")
    state2 = ppFPFState(fpf)
    print("-")
    @test no_of_particles(state2) == 10
    print("-")
    @test all(0 .<= state2.ensemble.positions .<= 2π)
    filt_prob = FilteringProblem(st_mod, ob_mod)
    state3 = ppFPFState(filt_prob, 20)
    print("-")
    @test no_of_particles(state3) == 20
    println("DONE")
    
    print("  outer constructors for ppFPF")
    fpf2 = ppFPF(filt_prob, g_method, f_method, 100)
    print("-")
    @test no_of_particles(fpf2) == 100
    println("DONE")

    print("  method initialize")
    state4 = initialize(fpf)
    print("-")
    @test state4 isa ppFPFState
    print("-")
    @test all(0 .<= state4.ensemble.positions .<= 2π)
    println("DONE")
    
    print("  method assimilate!")
    oldstate = deepcopy(state)
    assimilate!([0, 1, 0, 1], state, fpf, 0.01)
    print("-")
    @test state.eq_dt.gain != oldstate.eq_dt.gain
    solve!(oldstate.eq_dt, g_method)
    @test state.eq_dt.gain ≈ oldstate.eq_dt.gain
    
    print("-")
    @test state.eq_dt.positions ≈ state.ensemble.positions
    print("-")
    @test state.ensemble != oldstate.ensemble
    println("DONE")
    
    print("  method propagate!")
    oldstate = deepcopy(state)
    propagate!(state, fpf, 0.01)
    print("-")
    @test all(state.ensemble.positions .!= oldstate.ensemble.positions)
    println("DONE")
    
    print("  method update!")
    state1 = initialize(fpf) 
    state2 = deepcopy(state1)
    Random.seed!(0)
    update!(state1, fpf, [1, 1, 0, 0], 0.01)
    Random.seed!(0)
    propagate!(state2, fpf, 0.01)
    assimilate!([1, 1, 0, 0], state2, fpf, 0.01)
    print("-")
    @test state1.ensemble.positions ≈ state2.ensemble.positions
    print("-")
    @test state1.eq_dt.gain ≈ state2.eq_dt.gain
    println("DONE")
end; #ppFPF.jl
