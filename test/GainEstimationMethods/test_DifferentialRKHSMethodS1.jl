using FeedbackParticleFilters, Distributions, Random, PDMats

println("Testing DifferentialRKHSMethodS1.jl:")

@testset "DifferentialRKHSMethodS1.jl" begin
    print("  solver")
    N = 6
    x_pf=[-2.422480820086937 -0.592332203303167 -2.017301296096984 -1.5151245392598531 0.02565906919199346 0.15161614796874012;]
    testens=UnweightedParticleEnsemble(x_pf)
    eq = PoissonEquation(x->[x[1]^2,exp(x[1])], testens)
    update!(eq, testens)
    solve!(eq, DifferentialRKHSMethodS1(1E1, 1E-6));
    print("-")
    @test maximum(abs.(eq.gain[1,:,1] - [ -0.03630508403333434
                                          -0.5438751742317391
                                          -1.0013005320529875
                                          -0.8498038869177289
                                          -1.0778306319564457
                                           1.0096426629620914 ])) < 1e-6
    print("-")
    @test maximum(abs.(eq.gain[1,:,2] - [ -0.04446454607643974
                                           0.15511332541765455
                                           0.0646397390242831
                                           0.10240796498343294
                                           0.35929499989174624
                                          -0.2929181164799246 ])) < 1e-6
    print("-")
    @test maximum(abs.(eq.potential[1,:] - [ 96.14908939499902
                                           -140.78275506175999
                                             56.7520800164404
                                            122.01620383184002
                                            -66.39285991954131
                                            -67.74175826197815 ])) < 1e-6
    print("-")
    @test maximum(abs.(eq.potential[2,:] - [ -17.371718290088566
                                              24.208561463587788
                                              -9.588138488155298
                                             -23.06405207506813
                                              12.786152526976196
                                              13.02919486274801 ])) < 1e-6
    println("DONE")
end#DifferentialRKHSMethodS1.jl
