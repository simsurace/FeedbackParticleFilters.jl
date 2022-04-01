using FeedbackParticleFilters, Statistics

println("Testing ConstantGainApproximation.jl:")

@testset "ConstantGainApproximation.jl" begin
    
    h(x)      = x[1:2].^2
    pos       = randn(3, 10)
    eq        = PoissonEquation(h, pos)
    solve!(eq, ConstantGainApproximation())
    
    print("  solver")
    for i in 2:10, j in 1:2
        print("-")
        @test eq.gain[:,1,j] == eq.gain[:,i,j]
    end
    for i in 1:3, j in 1:2
        print("-")
        @test eq.gain[i,1,j] == Statistics.mean(eq.positions[i,:] .* (eq.H[j,:] .- eq.mean_H[j,:]))
    end
    println("DONE")
    
    
end; #ConstantGainApproximation.jl
