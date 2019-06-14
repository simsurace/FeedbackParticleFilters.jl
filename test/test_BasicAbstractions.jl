using FeedbackParticleFilters

println("Testing basic abstractions:")
@testset "Hidden state" begin
    print("  Hidden state")
    print(".")
    @test Int <: AbstractHiddenState
    print(".")
    @test Float64 <: AbstractHiddenState
    print(".")
    @test Vector{Int} <: AbstractHiddenState
    print(".")
    @test Vector{Float64} <: AbstractHiddenState
    print(".")
    @test !(Matrix{Float64} <: AbstractHiddenState)
    print(".")
    @test VectorHiddenState <: AbstractHiddenState
    println("DONE.")
end; #Hidden state

@testset "Filter representations" begin
    print("  Filter representations")
    print(".")
    @test ParticleRepresentation{Float64} <: AbstractFilterRepresentation{Float64}
    print(".")
    @test ParticleRepresentation{Int} <: AbstractFilterRepresentation{Int}
    print(".")
    @test UnweightedParticleRepresentation{Float64} <: ParticleRepresentation{Float64}
    print(".")
    @test UnweightedParticleRepresentation{Int} <: ParticleRepresentation{Int}
    println("DONE.")
end; #Filter representations

@testset "Gain equation" begin
    print("  Gain equation")
    print(".")
    @test EmptyGainEquation{Float64} <: AbstractGainEquation{Float64}
    print(".")
    @test isa(EmptyGainEquation{Float64}(), AbstractGainEquation{Float64})
    println("DONE.")
end; #Gain equation