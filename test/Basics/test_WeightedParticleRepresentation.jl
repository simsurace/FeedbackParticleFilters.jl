using FeedbackParticleFilters

println("Testing WeightedParticleRepresentation.jl:")

@testset "BPF.jl" begin
    
    struct TestRep{Float} <: WeightedParticleRepresentation{Vector{Float}}
        positions::Matrix{Float}
        weights::Vector{Float}
    end
    
    function FeedbackParticleFilters.get_weight(ens::TestRep, i)
        return ens.weights[i]
    end
    
    function FeedbackParticleFilters.no_of_particles(ens::TestRep)
        return 5
    end

    print("  method get_weight")
    pos     = randn(1, 5)
    weights = [0.2, 0.2, 0.2, 0.2, 0.2]
    ens     = TestRep(pos, weights)
    for i in 1:5
        print(".")
        @test get_weight(ens, i) == ens.weights[i]
    end
    println("DONE.")
        
    print("  method list_of_weights")
    print(".")
    @test list_of_weights(ens) == weights
    println("DONE.")
    
    print("  method sum_of_weights")
    print(".")
    @test sum_of_weights(ens) == 1.
    println("DONE.")
    
    print("  method eff_no_of_particles")
    print(".")
    @test eff_no_of_particles(ens) ≈ 5
    weights = [1., 0., 0., 0., 0.]
    ens     = TestRep(pos, weights)
    print(".")
    @test eff_no_of_particles(ens) ≈ 1
    weights = [.5, .5, 0., 0., 0.]
    ens     = TestRep(pos, weights)
    print(".")
    @test eff_no_of_particles(ens) ≈ 2
    println("DONE.")
    
    
end #WeightedParticleRepresentation.jl