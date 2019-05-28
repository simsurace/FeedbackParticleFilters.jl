push!(LOAD_PATH,"./")
print("Importing package PFlib...")
using PFlib # import particle filtering library
println("DONE")
print("Importing other packages...")
using Test
using StatsBase
println("DONE")

@testset "All tests" begin
@testset "basic functions" begin
    println("Basic function tests:")
    @testset "unweighted particle ensemble" begin
        print("  unweighted particle ensemble")
        testens = UnweightedParticleEnsemble([-2.422480820086937
                                              -0.592332203303167
                                              -2.017301296096984
                                              -1.5151245392598531
                                               0.02565906919199346
                                               0.15161614796874012],6,zeros(Float64,6),ones(Float64,6));
        print(".")
        @test typeof(testens.positions) == Array{Float64,1}
        print(".")
        @test typeof(testens.size) == Int
        print(".")
        @test typeof(testens.gain) == Array{Float64,1}
        print(".")
        @test typeof(testens.potential) == Array{Float64,1}
        println("DONE")
    end
        
    @testset "weighted particle ensemble" begin
        print("  weighted particle ensemble")
        testens = WeightedParticleEnsemble([-2.422480820086937
                                              -0.592332203303167
                                              -2.017301296096984
                                              -1.5151245392598531
                                               0.02565906919199346
                                               0.15161614796874012],
                                           ProbabilityWeights([0.5584525677704708   
                                            0.935086525152526    
                                            0.6482776466243567   
                                            0.16092470299756667  
                                            0.0024994760081029632
                                            0.7064569073485585]),6);
        print(".")
        @test typeof(testens.positions) == Array{Float64,1}
        print(".")
        @test typeof(testens.size) == Int
        print(".")
        @test typeof(testens.weights) == ProbabilityWeights{Float64,Float64,Array{Float64,1}}
        println("DONE")
    end
end;







@testset "FPF gain estimation" begin
    println("FPF gain estimation tests:")
    @testset "1d semigroup gain estimation" begin
        print("  1d semigroup method")
        testens = UnweightedParticleEnsemble([-2.422480820086937
                                              -0.592332203303167
                                              -2.017301296096984
                                              -1.5151245392598531
                                               0.02565906919199346
                                               0.15161614796874012],6,zeros(Float64,6),ones(Float64,6));
        print(".")
        Gain_semigroup!(testens);
        @test maximum(abs.(testens.gain -   [0.2693220783443806 
                                             1.976582040902698  
                                             0.5635782390559818 
                                             1.8205324118211719 
                                             0.37877152214068527
                                             0.2722780175889505])) < 1E-9
        print(".")
        Gain_semigroup!(testens);
        @test maximum(abs.(testens.gain -   [0.2701824403154385 
                                             1.9843317728814558 
                                             0.5655095454525805 
                                             1.8276468776242472 
                                             0.38012021882405683
                                             0.27323983784928857])) < 1E-9
        print(".")
        Gain_semigroup!(testens, x->exp(x));
        @test maximum(abs.(testens.gain -   [0.1202658249478139 
                                             1.1421290910557087 
                                             0.26803141265931085
                                             0.9805223682749997 
                                             0.2346199723013512 
                                             0.1696150787488252])) < 1E-9
        println("DONE")
    end
end;
end;