using FeedbackParticleFilters, StatsBase, Distributions

println("Testing particle ensembles:")
@testset "FPFEnsemble" begin
    print("  FPFEnsemble")
    print(".")
    @test FPFEnsemble <: UnweightedParticleRepresentation
    print(".")
    @test_throws ErrorException FPFEnsemble([1,2], 3)
    print(".")
    @test length(FPFEnsemble([1], 3).positions) == 3
    print(".")
    @test FPFEnsemble([1], 3).size == 3
    print(".")
    @test length(FPFEnsemble([1,2,3], 3).positions) == 3
    print(".")
    @test FPFEnsemble([1,2,3], 3).size == 3
    print(".")
    @test length(FPFEnsemble(Normal(), 3).positions) == 3
    print(".")
    @test FPFEnsemble(Normal(), 3).size == 3
    println("DONE.")
end; #FPFEnsemble