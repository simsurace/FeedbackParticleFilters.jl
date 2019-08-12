using FeedbackParticleFilters, Statistics

println("Testing PoissonEquation.jl:")

@testset "PoissonEquation.jl" begin
    
    h(x)      = 2x[1:2]
    pos       = randn(3, 10)
    H         = randn(2, 10)
    mean_H    = Statistics.mean(H, dims=2)
    potential = zeros(2, 10)
    gain      = zeros(3, 10, 2);
    
    print("  inner constructor")
    eq = PoissonEquation(h, pos, H, mean_H, potential, gain)
    print(".")
    @test eq isa PoissonEquation{Float64,typeof(h)}
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, randn(2, 10), H, mean_H, potential, gain) # wrong size of pos
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, randn(3, 11), H, mean_H, potential, gain) # wrong size of pos
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, randn(3,10), mean_H, potential, gain) # wrong size of H
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, randn(2,11), mean_H, potential, gain) # wrong size of H
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, H, randn(3,1), potential, gain) # wrong size of mean_H
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, H, randn(2,2), potential, gain) # wrong size of mean_H
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, H, mean_H, randn(3,10), gain) # wrong size of potential
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, H, mean_H, randn(2,11), gain) # wrong size of potential
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, H, mean_H, potential, randn(2,10,2)) # wrong size of gain
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, H, mean_H, potential, randn(3,11,2)) # wrong size of gain
    print(".")
    @test_throws DimensionMismatch PoissonEquation(h, pos, H, mean_H, potential, randn(3,10,3)) # wrong size of gain
    println("DONE.")
    
    print("  outer constructor")
    eq2 = PoissonEquation(h, pos)
    print(".")
    @test eq2 isa PoissonEquation{Float64,typeof(h)}
    print(".")
    @test eq2.H == 2*pos[1:2, :]
    print(".")
    @test eq2.mean_H == Statistics.mean(eq2.H, dims=2)
    ens = UnweightedParticleEnsemble(pos)
    eq3 = PoissonEquation(h, ens)
    print(".")
    @test eq3.positions == pos
    print(".")
    @test eq3.H == eq2.H
    print(".")
    @test eq3.mean_H == eq2.mean_H
    println("DONE.")
    
    print("  method state_dim")
    print(".")
    @test state_dim(eq) == 3
    println("DONE.") 
    
    print("  method obs_dim")
    print(".")
    @test obs_dim(eq) == 2
    println("DONE.") 
    
    print("  method Htilde")
    print(".")
    @test Htilde(eq2) == eq2.H .- eq2.mean_H
    println("DONE.") 
    
    print("  method update!")
    update!(eq)
    print(".")
    @test eq.positions == pos
    print(".")
    @test eq.H == 2*pos[1:2, :]
    print(".")
    @test eq.mean_H == Statistics.mean(eq.H, dims=2)
    pos = randn(3, 10)
    update!(eq, pos)
    print(".")
    @test eq.positions == pos
    print(".")
    @test eq.H == 2*pos[1:2, :]
    print(".")
    @test eq.mean_H == Statistics.mean(eq.H, dims=2)
    pos = randn(3, 10)
    ens = UnweightedParticleEnsemble(pos)
    update!(eq, ens)
    print(".")
    @test eq.positions == pos
    print(".")
    @test eq.H == 2*pos[1:2, :]
    print(".")
    @test eq.mean_H == Statistics.mean(eq.H, dims=2)
    println("DONE.")
    
    
end; #PoissonEquation.jl