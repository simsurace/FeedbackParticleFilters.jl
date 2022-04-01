using FeedbackParticleFilters, Distributions, Random, PDMats

println("Testing LinearDiffusionStateModel.jl:")

@testset "LinearDiffusionStateModel.jl" begin
    
    A      = [0. 1.; -0.7 -0.3]
    B      = zeros(2,1)
    B[2,1] = sqrt(2.)
    init   = MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1., 2)))
    
    print("  inner constructor")
    model = LinearDiffusionStateModel(A, B, init)
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
    @test drift(model) == A
    println("DONE.")
    
    print("  method diffusion")
    print(".")
    @test diffusion(model) == B
    println("DONE.")
    
    print("  method initial_condition")
    print(".")
    @test initial_condition(model) == model.init
    println("DONE.")
    
    print("  method initialize")
    print(".")
    x0 = initialize(model)
    @test length(x0) == 2
    println("DONE.")
    
    print("  method state_dim")
    print(".")
    @test state_dim(model) == 2
    println("DONE.")
    
    print("  method noise_dim")
    print(".")
    @test noise_dim(model) == 1
    println("DONE.")
    
    print("  method drift_function")
    print(".")
    @test drift_function(model)([1., 3.]) == [3.0, -1.5999999999999999]
    println("DONE.")
    
    print("  method diffusion_function")
    print(".")
    @test diffusion_function(model)([1., 3.]) == hcat([0.0, 1.4142135623730951]...)'
    println("DONE.")
    
    print("  calling instance of DiffusionStateModel")
    print(".")
    x0 = ones(2)
    @test model(x0, 0.01) != x0
    print(".")
    xx0 = hcat(ones(2), zeros(2))
    @test model(xx0, 0.01) != xx0
    println("DONE.")
    
    print("  method propagate!")
    print(".")
    x1 = copy(x0)
    propagate!(x1, model, 0.01)
    @test x1 != x0
    print(".")
    xx1 = copy(xx0)
    propagate!(xx1, model, 0.01)
    @test xx1 != xx0
    println("DONE.")
    
end; #LinearDiffusionStateModel.jl
