using FeedbackParticleFilters, StatsBase, Test

println("Testing gain equations:")
@testset "Scalar Poisson equation" begin
    print("  Scalar Poisson equation")
    N = 5
    eq = ScalarPoissonEquation(x->x, N)
    print(".") 
    @test length(eq.positions) == N
    print(".") 
    @test length(eq.H) == N
    print(".") 
    @test length(eq.potential) == N
    print(".") 
    @test length(eq.gain) == N
    pos = rand(N)
    testens=FPFEnsemble(pos,N)
    Update!(eq, testens)
    print(".") 
    @test eq.positions == pos
    print(".")
    @test eq.H == pos
    print(".")
    @test eq.mean_H == StatsBase.mean(pos)
    eq.h = x->x^2
    Update!(eq, testens)
    print(".")
    @test eq.H == pos.^2
    print(".")
    @test eq.mean_H == StatsBase.mean(pos.^2)
    println("DONE.")
end; #Scalar Poisson equation

@testset "Vector-scalar Poisson equation" begin
    print("  Vector-scalar Poisson equation")
    println("DONE.")
end; #Vector-scalar Poisson equation

@testset "Scalar-vector Poisson equation" begin
    print("  Scalar-vector Poisson equation")
    println("DONE.")
end; #Scalar-vector Poisson equation

@testset "Vector-vector Poisson equation" begin
    print("  Vector-vector Poisson equation")
    println("DONE.")
end; #Vector-vector Poisson equation
