using FeedbackParticleFilters, Distributions, Random, PDMats

println("Testing DifferentialRKHSMethodS1.jl:")

@testset "DifferentialRKHSMethodS1.jl" begin
    print("  solver")
    N = 6
    x_pf=[0.3226665452501024 0.33253916437044034 0.3888772591442652 0.4612018638628069 0.9927640673696214 5.819227805336827;]
    testens=UnweightedParticleEnsemble(x_pf)
    eq = PoissonEquation(x->[x[1],exp(x[1])], testens)
    update!(eq, testens)
    solve!(eq, DifferentialRKHSMethod(1E1, 1E-6));
    print(".")
    @test maximum(abs.(eq.gain[1,:,1] - [ 0.1312360106492137 
                                       2.343121868385176  
                                       0.863411396927212  
                                       1.9307957511303933 
                                       0.5684889795489233 
                                       0.00768453173042769 ])) < 1e-6
    print(".")
    @test maximum(abs.(eq.potential[1,:] - [ -2.072323421422226 
                                             1.0382735928503166
                                            -1.8816738883083453
                                            -1.1749266092106723
                                             2.0268744617672745
                                             2.0637758643236275 ])) < 1e-6
    println("DONE.")
end#DifferentialRKHSMethodS1.jl