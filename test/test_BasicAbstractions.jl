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

@testset "Models" begin
    print("  Models")
    print(".")
    @test HiddenStateModel{Float64} <: AbstractModel{Float64}
    struct StMod{T} <: HiddenStateModel{T} end
    struct ObsMod{S, T} <: ObservationModel{S, T} end
    struct FiltProb{S, T, M1, M2} <: AbstractFilteringProblem{S, T}
        state_model::M1
        obs_model::M2
        FiltProb(mod1::HiddenStateModel{S}, mod2::ObservationModel{S,T}) where {S, T} = new{S, T, typeof(mod1), typeof(mod2)}(mod1, mod2)
    end
    state_model = StMod{Float64}();
    obs_model = ObsMod{Float64, Float64}();
    filt_prob = FiltProb(state_model, obs_model)
    print(".")
    @test isa(filt_prob, AbstractFilteringProblem{Float64,Float64})
    obs_model = ObsMod{Int, Float64}();
    print(".")
    @test_throws MethodError FilteringProblem(state_model, obs_model)
    println("DONE.")
end; #Models

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