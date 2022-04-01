using FeedbackParticleFilters, Distributions, LinearAlgebra, Random, PDMats, FillArrays

println("Testing BrownianMotionTorus.jl:")

@testset "BrownianMotionTorus.jl" begin
    @testset "$(d)d" for d in 1:3
        print("  inner constructor")
        model = BrownianMotionTorus(d)
        
        print("  method state_dim")
        print("-")
        @test state_dim(model) == d
        println("DONE")
        
        print("  method noise_dim")
        print("-")
        @test noise_dim(model) == d
        println("DONE")
        
        print("  method initialize")
        x0 = initialize(model)
        print("-")
        @test all(0 .<= x0 .<= 2π)
        println("DONE")
        
        print("  calling instance of BrownianMotionTorus")
        x0 = ones(d)
        x1 = model(x0, 0.01)
        print("-")
        @test x1 != x0
        print("-")
        @test all(0 .<= x0 .<= 2π)
        xx0 = hcat(ones(d), zeros(d))
        xx1 = model(xx0, 0.01)
        print("-")
        @test xx1 != xx0
        print("-")
        @test all(0 .<= x0 .<= 2π)
        println("DONE")
        
        print("  method propagate!")
        x0 = ones(d)
        x1 = copy(x0)
        propagate!(x1, model, 0.01)
        print("-")
        @test x1 != x0
        xx0 = hcat(ones(d), zeros(d))
        xx1 = copy(xx0)
        propagate!(xx1, model, 0.01)
        print("-")
        @test xx1 != xx0
        println("DONE")
    end
end; #BrownianMotionTorus.jl
