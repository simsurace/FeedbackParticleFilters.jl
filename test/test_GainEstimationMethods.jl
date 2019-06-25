using FeedbackParticleFilters, Test, StatsBase

println("Testing gain estimation methods:")
@testset "1d semigroup gain estimation" begin
        print("  1d semigroup method")
        N = 6
        eq = ScalarPoissonEquation(x->x, randn(N), zeros(Float64,N), 0., ones(Float64,N), zeros(Float64,N))
        x_pf=[-2.422480820086937
              -0.592332203303167
              -2.017301296096984
              -1.5151245392598531
               0.02565906919199346
               0.15161614796874012]
        testens=FPFEnsemble(x_pf,6)
        Update!(eq, testens)
        Solve!(eq, SemigroupMethod1d(0.1,0.01));
        print(".")
        @test maximum(abs.(eq.gain -   [0.2693220783443806 
                                             1.976582040902698  
                                             0.5635782390559818 
                                             1.8205324118211719 
                                             0.37877152214068527
                                             0.2722780175889505])) < 1E-9
        Solve!(eq, SemigroupMethod1d(0.1,0.01));
        print(".")
        @test maximum(abs.(eq.gain -   [0.2701824403154385 
                                             1.9843317728814558 
                                             0.5655095454525805 
                                             1.8276468776242472 
                                             0.38012021882405683
                                             0.27323983784928857])) < 1E-9
        eq.h = x->exp(x)
        Update!(eq, testens)
        Solve!(eq, SemigroupMethod1d(0.1,0.01));
        print(".")
        @test maximum(abs.(eq.gain -   [0.1202658249478139 
                                             1.1421290910557087 
                                             0.26803141265931085
                                             0.9805223682749997 
                                             0.2346199723013512 
                                             0.1696150787488252])) < 1E-9
        println("DONE.")
end #testset 1d semigroup gain estimation
    
@testset "1d regularized semigroup gain estimation" begin
        print("  1d regularized semigroup method")
        N = 6
        eq = ScalarPoissonEquation(x->x, randn(N), zeros(Float64,N), 0., ones(Float64,N), zeros(Float64,N))
        x_pf=[-2.422480820086937
              -0.592332203303167
              -2.017301296096984
              -1.5151245392598531
               0.02565906919199346
               0.15161614796874012]
        testens=FPFEnsemble(x_pf,6)
        Update!(eq, testens)
        Solve!(eq, RegularizedSemigroupMethod1d(0.1,0.01,100,1E-3));
        print(".")
        @test maximum(abs.(eq.gain -   [0.28933745443135983,
                                         1.931177022246619,
                                         0.572385061676326,  
                                         1.794729522993188,  
                                         0.37094867389612113,
                                         0.26693841191026996])) < 1E-2
        Solve!(eq, RegularizedSemigroupMethod1d(0.1,0.01,100,1E-3));
        print(".")
        @test maximum(abs.(eq.gain -   [0.28933745443135983,
                                         1.931177022246619,
                                         0.572385061676326,  
                                         1.794729522993188,  
                                         0.37094867389612113,
                                         0.26693841191026996])) < 1E-2
        eq.h = x->exp(x)
        Update!(eq, testens)
        Solve!(eq, RegularizedSemigroupMethod1d(0.1,0.01,100,1E-3));
        print(".")
        @test maximum(abs.(eq.gain -   [0.13017718229847874,
                                         1.1152336243234484 ,
                                         0.2732138843889431 ,
                                         0.9652901439352324 ,
                                         0.22998346900674047,
                                         0.16644355667591126])) < 1E-2
        println("DONE.")
end #testset 1d regularized semigroup gain estimation

@testset "1d differential loss RKHS gain estimation" begin
        print("  1d differential loss RKHS method")
        N = 6
        eq = ScalarPoissonEquation(x->x, randn(N), zeros(Float64,N), 0., ones(Float64,N), zeros(Float64,N))
        x_pf=[-2.422480820086937
              -0.592332203303167
              -2.017301296096984
              -1.5151245392598531
               0.02565906919199346
               0.15161614796874012]
        testens=FPFEnsemble(x_pf,N)
        Update!(eq, testens)
        Solve!(eq, DifferentialRKHSMethod1d(1E1, 1E-6));
        print(".")
        @test maximum(abs.(eq.gain - [ 0.1312360106492137 
                                       2.343121868385176  
                                       0.863411396927212  
                                       1.9307957511303933 
                                       0.5684889795489233 
                                       0.00768453173042769 ])) < 1e-6
        print(".")
        @test maximum(abs.(eq.potential - [ -2.072323421422226 
                                             1.0382735928503166
                                            -1.8816738883083453
                                            -1.1749266092106723
                                             2.0268744617672745
                                             2.0637758643236275 ])) < 1e-6
        println("DONE.")
end #testset 1d differential loss RKHS gain estimation

