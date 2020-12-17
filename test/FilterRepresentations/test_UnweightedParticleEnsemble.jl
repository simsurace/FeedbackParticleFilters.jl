using FeedbackParticleFilters, Statistics, StatsBase

println("Testing UnweightedParticleEnsemble.jl:")

@testset "UnweightedParticleEnsemble.jl" begin
    
    print("  inner constructor for UnweightedParticleEnsemble")
    N    = 6
    d    = 2
    x_pf = [-2.422480820086937 -0.592332203303167 -2.017301296096984 -1.5151245392598531 0.02565906919199346  0.15161614796874012;
            -1.4986273707574955 0.3549802657063476 0.3625127047102   -0.166475807288935  0.05739520762738308 -0.18730971131161062]
    ens  = UnweightedParticleEnsemble(x_pf)
    print(".")
    @test ens.positions == x_pf
    println("DONE.")
    
    print("  method dim")
    print(".")
    @test dim(ens) == d * N
    println("DONE.")
    
    print("  method particle_dim")
    print(".")
    @test particle_dim(ens) == d
    println("DONE.")
    
    print("  method no_of_particles")
    print(".")
    @test no_of_particles(ens) == N
    println("DONE.")
    
    print("  method eff_no_of_particles")
    print(".")
    @test eff_no_of_particles(ens) == N
    println("DONE.")
    
    print("  method get_pos")
    for i in 1:N
        print(".")
        @test get_pos(ens, i) == x_pf[:, i]
    end
    println("DONE.")
    
    print("  method mean")
    print(".")
    @test size(mean(ens)) == (d, 1)
    print(".")
    @test mean(ens)[:,1] == [-1.0616606069310346; -0.17958745188568506]
    println("DONE.")
    
    print("  method cov")
    print(".")
    @test size(cov(ens)) == (d, d)
    print(".")
    @test cov(ens) == [0.9742140049604432 0.29502867377601394; 0.29502867377601394 0.39598231024579167]
    println("DONE.")
    
    print("  method var")
    print(".")
    @test size(var(ens)) == (d, 1)
    print(".")
    @test var(ens)[:,1] == [0.9742140049604432; 0.39598231024579167]
    println("DONE.")
    
end; #UnweightedParticleEnsemble.jl