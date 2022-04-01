using FeedbackParticleFilters, Random, Statistics, LinearAlgebra

println("Testing KBF.jl:")

@testset "KBF.jl" begin
    
    A = randn(2,2)
    B = randn(2,3)
    C = randn(4,2)
    
    BB = B*B'
    CC = C'*C
    
    m = zeros(2)
    P = zeros(2,2)
    
    
    
    print("  inner constructor for KBState")
    init = KBState(MultivariateGaussian(m,P))
    print("-")
    @test typeof(init) == KBState{MultivariateGaussian{Float64,Array{Float64,1},Array{Float64,2}}}
    println("DONE")
    
    print("  inner constructor for KBF")
    kbf = KBF(A, BB, C, CC, init)
    print("-")
    @test typeof(kbf) == KBF{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},KBState{MultivariateGaussian{Float64,Array{Float64,1},Array{Float64,2}}}}
    print("-")
    @test_throws DimensionMismatch KBF(randn(1,2), BB, C, CC, init)
    println("DONE")
    
    print("  method initial_condition")
    print("-")
    @test initial_condition(kbf) == kbf.init
    println("DONE")
    
    print("  method for Statistics.mean")
    print("-")
    @test Statistics.mean(init) == init.gauss.mean
    println("DONE")
    
    print("  method for Statistics.cov")
    print("-")
    @test Statistics.cov(init) == init.gauss.cov
    println("DONE")
    
    print("  method for Statistics.var")
    print("-")
    @test Statistics.var(init) == LinearAlgebra.diag(init.gauss.cov)
    println("DONE")
    
    print("  outer constructors for KBState")
    init2 = KBState(m, P)
    print("-")
    @test init2 == init
    init3 = KBState(3)
    print("-")
    @test iszero(init3.gauss.mean)
    print("-")
    @test iszero(init3.gauss.cov)
    
    
    
    A      = [0. 1.; -0.7 -0.3]
    B      = zeros(2,1)
    B[2,1] = sqrt(2.)
    C      = [1. 2. ; 2. 3. ; 3. 4.]
    BB = B*B'
    CC = C'*C
    st_mod = LinearDiffusionStateModel(A, B)
    ob_mod = LinearDiffusionObservationModel(C)
    f_prob = FilteringProblem(st_mod, ob_mod)
    
    
    init4  = KBState(f_prob)
    print("-")
    @test init4.gauss.mean == [0.0, 0.0]
    print("-")
    @test init4.gauss.cov == [4.746796851882334 -5.543824348282954e-5; -5.543824348282954e-5 3.3192089048776765]
    println("DONE")
    
    print("  outer constructors for KBF")
    kbf2 = KBF(A, B, C, init)
    print("-")
    @test kbf2.BB == BB
    print("-")
    @test kbf2.CC == CC
    print("-")
    @test kbf2.init == init
    
    kbf3 = KBF(A, B, C)
    print("-")
    @test typeof(kbf3) == KBF{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},KBState{MultivariateGaussian{Float64,Array{Float64,1},Array{Float64,2}}}}
    print("-")
    @test kbf3.BB == BB
    print("-")
    @test kbf3.CC == CC
    print("-")
    @test iszero(kbf3.init.gauss.mean)
    print("-")
    @test kbf3.init.gauss.cov ≈ [4.746796851882334 -5.543824348282954e-5; -5.543824348282954e-5 3.3192089048776765]
    
    kbf4 = KBF(f_prob)
    print("-")
    @test typeof(kbf4) == KBF{Array{Float64,2},Array{Float64,2},Array{Float64,2},Array{Float64,2},KBState{MultivariateGaussian{Float64,Array{Float64,1},Array{Float64,2}}}}
    print("-")
    @test kbf4.BB == BB
    print("-")
    @test kbf4.CC == CC
    print("-")
    @test iszero(kbf4.init.gauss.mean)
    print("-")
    @test kbf4.init.gauss.cov ≈ [4.746796851882334 -5.543824348282954e-5; -5.543824348282954e-5 3.3192089048776765]
    println("DONE")
    
    print("  method initialize")
    init5 = initialize(kbf4)
    print("-")
    @test init5 == kbf4.init
    println("DONE")
    
    print("  method update!")
    print("-")
    state = deepcopy(kbf3.init)
    update!(state, kbf3, [0.1, 0.2, 0.3], 0.01)
    print("-")
    @test state.gauss.mean ≈ [6.6454047161483025, 6.638340196214477 ]
    print("-")
    @test state.gauss.cov ≈ [1.5924097544324285 -3.151122632746998; -3.151122632746998 0.12439518287453177]
    println("DONE")
    
    
end; #KBF.jl
