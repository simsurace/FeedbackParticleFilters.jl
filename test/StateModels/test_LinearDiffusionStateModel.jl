using FeedbackParticleFilters, Distributions, Random, PDMats

println("Testing LinearDiffusionStateModel.jl:")

@testset "LinearDiffusionStateModel.jl" begin
    
    A      = [0. 1.; -0.7 -0.3]
    B      = zeros(2,1)
    B[2,1] = sqrt(2.)
    init   = MvNormal(2, 1.)
    
    print("  inner constructor")
    mod = LinearDiffusionStateModel(A, B, init)
    print(".")
    @test mod isa LinearDiffusionStateModel{Float64,Array{Float64,2},Array{Float64,2},MvNormal{Float64,PDMats.ScalMat{Float64},Distributions.ZeroVector{Float64}}}
    print(".")
    B2 = zeros(Int, 2, 1)
    B2[2,1] = 2
    @test_throws MethodError LinearDiffusionStateModel(A, B2, init) # wrong element type of B2
    print(".")
    B3 = zeros(3, 1)
    @test_throws DimensionMismatch LinearDiffusionStateModel(A, B3, init) # wrong size of B3
    println("DONE.")
    
    print("  method find_stationary_variance")
    print(".")
    Sigma = find_stationary_variance(A, B*B')
    @test Sigma == [4.746796851882334 -5.543824348282954e-5; -5.543824348282954e-5 3.3192089048776765]
    println("DONE.")
    
    print("  method drift")
    print(".")
    @test drift(mod) == A
    println("DONE.")
    
    print("  method diffusion")
    print(".")
    @test diffusion(mod) == B
    println("DONE.")
    
    print("  method initial_condition")
    print(".")
    @test initial_condition(mod) == mod.init
    println("DONE.")
    
    print("  method initialize")
    print(".")
    Random.seed!(0)
    @test initialize(mod) == [0.6791074260357777, 0.8284134829000359]
    println("DONE.")
    
    print("  method state_dim")
    print(".")
    @test state_dim(mod) == 2
    println("DONE.")
    
    print("  method noise_dim")
    print(".")
    @test noise_dim(mod) == 1
    println("DONE.")
    
    print("  method drift_function")
    print(".")
    @test drift_function(mod)([1., 3.]) == [3.0, -1.5999999999999999]
    println("DONE.")
    
    print("  method diffusion_function")
    print(".")
    @test diffusion_function(mod)([1., 3.]) == hcat([0.0, 1.4142135623730951]...)'
    println("DONE.")
    
    print("  calling instance of DiffusionStateModel")
    print(".")
    Random.seed!(0)
    x0 = ones(2)
    @test mod(x0, 0.01) == [1.01, 1.086040293220808]
    print(".")
    Random.seed!(0)
    xx = hcat(ones(2), zeros(2))
    @test mod(xx, 0.01) == [1.01 0. ; 1.086040293220808 0.1171553582769963]
    println("DONE.")
    
    print("  method propagate!")
    print(".")
    Random.seed!(0)
    propagate!(x0, mod, 0.01)
    @test x0 == [1.01, 1.086040293220808]
    print(".")
    Random.seed!(0)
    propagate!(xx, mod, 0.01)
    @test xx == [1.01 0. ; 1.086040293220808 0.1171553582769963]
    println("DONE.")
    
end; #LinearDiffusionStateModel.jl