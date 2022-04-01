using FeedbackParticleFilters

println("Testing FilteringProblem.jl:")

@testset "FilteringProblem.jl" begin
    
    struct TestStateModel <: HiddenStateModel{Int, ContinuousTime} end
    struct TestObservationModel <: ObservationModel{Int, Float64, DiscreteTime} end

    st_mod    = TestStateModel()
    ob_mod    = TestObservationModel()
    filt_prob = FilteringProblem(st_mod, ob_mod)
    
    print("  struct FilteringProblem")
    print("-")
    @test filt_prob isa FilteringProblem{Int64,Float64,ContinuousTime,DiscreteTime,TestStateModel,TestObservationModel}
    println("DONE")
    
    print("  method state_model")
    print("-")
    @test state_model(filt_prob) isa TestStateModel
    println("DONE")
    
    print("  method obs_model")
    print("-")
    @test obs_model(filt_prob) isa TestObservationModel
    println("DONE")
    
end; #FilteringProblem.jl
