using FeedbackParticleFilters, Distributions, Test

println("Testing hidden state models:")
@testset "Scalar diffusion model" begin
    print("  Scalar diffusion model")
    print(".")
    @test ScalarDiffusionStateModel <: DiffusionStateModel <: HiddenStateModel
    f(x::Float64) = x
    g(x::Float64) = 1.
    print(".")
    @test DiffusionStateModel(f, g, 0.) == ScalarDiffusionStateModel(f, g, 0.)
    print(".")
    @test DiffusionStateModel(f, g, 0., 1, 1) == ScalarDiffusionStateModel(f, g, 0.)
    model = DiffusionStateModel(f, g, 0.)
    print(".")
    @test FPFEnsemble(model, 3).positions == [0.0, 0.0, 0.0]
    print(".")
    @test DiffusionStateModel(f, g, Normal()) == ScalarDiffusionStateModel(f, g, Normal())
    print(".")
    @test DiffusionStateModel(f, g, Normal(), 1, 1) == ScalarDiffusionStateModel(f, g, Normal())
    model = DiffusionStateModel(f, g, Normal())
    print(".")
    @test length(FPFEnsemble(model, 3).positions) == 3
    print(".")
    @test typeof(Propagator(model, 0.01)(0.)) == Float64
    testens = FPFEnsemble(model, 3)
    print(".")
    @test typeof(Propagator(model, 0.01)(testens)) == Nothing
    ff(x::Int64) = x
    fff(x::Float64) = [x, x]
    gg(x::Int64) = 1.
    ggg(x::Float64) = [1., 1.]
    print(".")
    @test_throws ErrorException ScalarDiffusionStateModel(ff, g, 0.)
    print(".")
    @test_throws ErrorException ScalarDiffusionStateModel(f, gg, 0.)
    print(".")
    @test_throws ErrorException ScalarDiffusionStateModel(fff, g, 0.)
    print(".")
    @test_throws ErrorException ScalarDiffusionStateModel(f, ggg, 0.)
    println("DONE.")
end; #Scalar diffusion model

@testset "Vector diffusion model" begin
    print("  Vector diffusion model")
    println("DONE.")
end; #Vector diffusion model