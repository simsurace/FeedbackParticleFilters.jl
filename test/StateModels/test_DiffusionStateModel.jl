using FeedbackParticleFilters, Distributions, LinearAlgebra, Random, PDMats, FillArrays

println("Testing DiffusionStateModel.jl:")

@testset "DiffusionStateModel.jl" begin
    
    f(x::AbstractVector) = [x[1]*x[2], x[1]-x[3], x[1]+x[2]*x[3]]
    g(x::AbstractVector) = [x[1] x[2]; x[2] x[3]; 0. 1.]
    init                 = MvNormal(LinearAlgebra.Diagonal(FillArrays.Fill(1., 3)))
    
    print("  inner constructor")
    model = DiffusionStateModel(f, g, init)
    print(".")
    @test_throws BoundsError DiffusionStateModel(f, g, [1., 2.]) # wrong length of initial condition
    print(".")
    @test_throws ErrorException DiffusionStateModel(f, g, [1., 2., 3., 4.]) # wrong length of initial condition
    print(".")
    g2(x::AbstractVector) = [x[1] x[2]; x[2] x[3]]
    @test_throws ErrorException DiffusionStateModel(f, g2, init) # wrong size of diffusion term
    print(".")
    @test_throws ErrorException DiffusionStateModel(f, g, [0, 0, 0]) # wrong type of initial condition
    println("DONE.")
    
    print("  method drift")
    print(".")
    @test drift(model)([1., 2., 3.]) == [2., -2., 7.]
    print(".")
    @test hasmethod(drift(model), Tuple{AbstractMatrix})
    println("DONE.")
    
    print("  method diffusion")
    print(".")
    @test diffusion(model)([1., 2., 3.]) == [1. 2.; 2. 3.; 0. 1.]
    print(".")
    @test hasmethod(diffusion(model), Tuple{AbstractMatrix})
    println("DONE.")
    
    print("  method initial_condition")
    print(".")
    @test initial_condition(model) == model.init
    println("DONE.")
    
    print("  method state_dim")
    print(".")
    @test state_dim(model) == 3
    println("DONE.")
    
    print("  method noise_dim")
    print(".")
    @test noise_dim(model) == 2
    println("DONE.")
    
    print("  method initialize")
    print(".")
    print(".")
    model2 = DiffusionStateModel(f, g, [1., 2., 3.])
    @test initialize(model2) == [1., 2., 3.]
    println("DONE.")
    
    print("  method drift_function")
    print(".")
    @test drift_function(model)([1., 2., 3.]) == drift(model)([1., 2., 3.])
    println("DONE.")
    
    print("  method diffusion_function")
    print(".")
    @test diffusion_function(model)([1., 2., 3.]) == diffusion(model)([1., 2., 3.])
    println("DONE.")
    
    print("  calling instance of DiffusionStateModel")
    print(".")
    x0 = ones(3)
    @test model(x0, 0.01) != x0
    print(".")
    xx = hcat(ones(3), zeros(3))
    @test model(xx, 0.01) != xx
    println("DONE.")
    
    print("  method propagate!")
    print(".")
    x0 = ones(3)
    x1 = copy(x0)
    propagate!(x1, model, 0.01)
    @test x1 != x0
    print(".")
    xx0 = hcat(ones(3), zeros(3))
    xx1 = copy(xx0)
    propagate!(xx1, model, 0.01)
    @test xx1 != xx0
    println("DONE.")
    
    
    
end; #DiffusionStateModel.jl
