@testset "Gain equations" begin
    println("Gain equation tests:")
    @testset "Poisson equation" begin
        print("  Poisson equations")
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
        testens=FPFEnsemble(pos,N,ScalarObservationData(pos,mean(pos)),GainDataSemigroup1d(zeros(Float64,N),ones(Float64,N)))
        Update!(eq, testens)
        print(".") 
        @test eq.positions == pos
        print(".")
        @test eq.H == pos
        print(".")
        @test eq.mean_H == mean(pos)
        print(".")
        eq.h = x->x^2
        Update!(eq, testens)
        @test eq.H == pos.^2
        print(".")
        @test eq.mean_H == mean(pos.^2)
        println("DONE")
    end; #Poisson equation
end; #testset gain equations

@testset "FPF gain estimation" begin
    println("FPF gain estimation tests:")
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
        testens=FPFEnsemble(x_pf,6,ScalarObservationData(x_pf,mean(x_pf)),GainDataSemigroup1d(zeros(Float64,6),ones(Float64,6)))
        Update!(eq, testens)
        print(".")
        Solve!(eq, SemigroupMethod1d(0.1,0.01));
        @test maximum(abs.(eq.gain -   [0.2693220783443806 
                                             1.976582040902698  
                                             0.5635782390559818 
                                             1.8205324118211719 
                                             0.37877152214068527
                                             0.2722780175889505])) < 1E-9
        print(".")
        Solve!(eq, SemigroupMethod1d(0.1,0.01));
        @test maximum(abs.(eq.gain -   [0.2701824403154385 
                                             1.9843317728814558 
                                             0.5655095454525805 
                                             1.8276468776242472 
                                             0.38012021882405683
                                             0.27323983784928857])) < 1E-9
        eq.h = x->exp(x)
        Update!(eq, testens)
        print(".")
        Solve!(eq, SemigroupMethod1d(0.1,0.01));
        @test maximum(abs.(eq.gain -   [0.1202658249478139 
                                             1.1421290910557087 
                                             0.26803141265931085
                                             0.9805223682749997 
                                             0.2346199723013512 
                                             0.1696150787488252])) < 1E-9
        println("DONE")
        end #testset 1d semigroup gain estimation
end; #testset FPF gain estimation
