using FeedbackParticleFilters, Distributions, Random, PDMats

println("Testing SemigroupMethod.jl:")

@testset "SemigroupMethod.jl" begin
    print("  solver")
    x_pf=[-2.422480820086937 -0.592332203303167 -2.017301296096984 -1.5151245392598531 0.02565906919199346 0.15161614796874012;]
    testens=UnweightedParticleEnsemble(x_pf)
    eq = PoissonEquation(x->[x[1],exp(x[1])], testens)
    update!(eq, testens)
    solve!(eq, SemigroupMethod(0.1,0.01,100));
    print(".")
    @test abs(eq.potential[1,1] - -2.479601997482972) < 1E-9
    print(".")
    @test abs(eq.potential[1,2] - 1.7209004619411277) < 1E-9
    print(".")
    @test abs(eq.potential[1,3] - -2.275929639206152) < 1E-9
    print(".")
    @test abs(eq.potential[1,4] - -1.7270260296018782) < 1E-9
    print(".")
    @test abs(eq.potential[1,5] - 2.354258246950214) < 1E-9
    print(".")
    @test abs(eq.potential[1,6] - 2.4073989573996615) < 1E-9
    print(".")
    @test abs(eq.gain[1,1,1] - 0.2693220783443806) < 1E-9
    print(".")
    @test abs(eq.gain[1,2,1] - 1.976582040902698) < 1E-9
    print(".")
    @test abs(eq.gain[1,3,1] - 0.5635782390559818) < 1E-9
    print(".")
    @test abs(eq.gain[1,4,1] - 1.8205324118211719) < 1E-9
    print(".")
    @test abs(eq.gain[1,5,1] - 0.37877152214068527) < 1E-9
    print(".")
    @test abs(eq.gain[1,6,1] - 0.2722780175889505) < 1E-9
    print(".")
    @test abs(eq.potential[2,1] - -1.0121206801162257) < 1E-9
    print(".")
    @test abs(eq.potential[2,2] - 0.6859801423517747) < 1E-9
    print(".")
    @test abs(eq.potential[2,3] - -0.9514708484563714) < 1E-9
    print(".")
    @test abs(eq.potential[2,4] - -0.75223951474933) < 1E-9
    print(".")
    @test abs(eq.potential[2,5] - 0.9979730732996036) < 1E-9
    print(".")
    @test abs(eq.potential[2,6] - 1.0318778276705487) < 1E-9
    print(".")
    @test abs(eq.gain[1,1,2] - 0.0899649944564862) < 1E-9
    print(".")
    @test abs(eq.gain[1,2,2] - 0.8691936023467642) < 1E-9
    print(".")
    @test abs(eq.gain[1,3,2] - 0.20001331130430455) < 1E-9
    print(".")
    @test abs(eq.gain[1,4,2] - 0.7299601286786728) < 1E-9
    print(".")
    @test abs(eq.gain[1,5,2] - 0.18712062788471728) < 1E-9
    print(".")
    @test abs(eq.gain[1,6,2] - 0.135741019165256) < 1E-9
    println("DONE.")
end#SemigroupMethod.jl