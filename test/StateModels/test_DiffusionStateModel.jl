using FeedbackParticleFilters, Distributions, Random, PDMats

println("Testing DiffusionStateModel.jl:")

@testset "DiffusionStateModel.jl" begin
    
    f(x::AbstractVector) = [x[1]*x[2], x[1]-x[3], x[1]+x[2]*x[3]]
    g(x::AbstractVector) = [x[1] x[2]; x[2] x[3]; 0. 1.]
    init                 = MvNormal(3, 1.)
    
    print("  inner constructor")
    mod = DiffusionStateModel(f, g, init)
    print(".")
    @test mod isa DiffusionStateModel{Float64,typeof(f),typeof(g),MvNormal{Float64,PDMats.ScalMat{Float64},Distributions.ZeroVector{Float64}}}
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
    @test drift(mod)([1., 2., 3.]) == [2., -2., 7.]
    print(".")
    @test hasmethod(drift(mod), Tuple{AbstractMatrix})
    println("DONE.")
    
    print("  method diffusion")
    print(".")
    @test diffusion(mod)([1., 2., 3.]) == [1. 2.; 2. 3.; 0. 1.]
    print(".")
    @test hasmethod(diffusion(mod), Tuple{AbstractMatrix})
    println("DONE.")
    
    print("  method initial_condition")
    print(".")
    @test initial_condition(mod) == mod.init
    println("DONE.")
    
    print("  method state_dim")
    print(".")
    @test state_dim(mod) == 3
    println("DONE.")
    
    print("  method noise_dim")
    print(".")
    @test noise_dim(mod) == 2
    println("DONE.")
    
    print("  method initialize")
    print(".")
    Random.seed!(0)
    @test initialize(mod) == [0.6791074260357777, 0.8284134829000359, -0.3530074003005963]
    print(".")
    mod2 = DiffusionStateModel(f, g, [1., 2., 3.])
    @test initialize(mod2) == [1., 2., 3.]
    println("DONE.")
    
    print("  method drift_function")
    print(".")
    @test drift_function(mod)([1., 2., 3.]) == drift(mod)([1., 2., 3.])
    println("DONE.")
    
    print("  method diffusion_function")
    print(".")
    @test diffusion_function(mod)([1., 2., 3.]) == diffusion(mod)([1., 2., 3.])
    println("DONE.")
    
    print("  calling instance of DiffusionStateModel")
    print(".")
    Random.seed!(0)
    x0 = ones(3)
    @test mod(x0, 0.01) == [1.1607520908935813, 1.1507520908935813, 1.1028413482900037]
    print(".")
    Random.seed!(0)
    xx = hcat(ones(3), zeros(3))
    @test mod(xx, 0.01) == [2.0744253554105256 0. ; 2.1415030557533146 0. ; 2.0697335850849417 -0.07628038164104582]
    println("DONE.")
    
    print("  method propagate!")
    print(".")
    Random.seed!(0)
    x0 = ones(3)
    propagate!(x0, mod, 0.01)
    @test x0 == [1.0744253554105254, 1.1415030557533146, 1.0697335850849417]
    print(".")
    Random.seed!(0)
    xx = hcat(ones(3), zeros(3))
    propagate!(xx, mod, 0.01)
    @test xx == [1.0744253554105254 0. ; 1.1415030557533146 0. ; 1.0697335850849417 -0.07628038164104582]
    println("DONE.")
    
    
    
end; #DiffusionStateModel.jl