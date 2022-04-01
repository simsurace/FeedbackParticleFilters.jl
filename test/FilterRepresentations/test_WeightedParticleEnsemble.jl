using FeedbackParticleFilters, Statistics, StatsBase

println("Testing WeightedParticleEnsemble.jl:")

@testset "WeightedParticleEnsemble.jl" begin
    
    print("  inner constructor for WeightedParticleEnsemble")
    N    = 6
    d    = 2
    x_pf = [-2.422480820086937 -0.592332203303167 -2.017301296096984 -1.5151245392598531 0.02565906919199346  0.15161614796874012;
            -1.4986273707574955 0.3549802657063476 0.3625127047102   -0.166475807288935  0.05739520762738308 -0.18730971131161062]
    w_pf = StatsBase.ProbabilityWeights([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
    ens  = WeightedParticleEnsemble(x_pf, w_pf)
    print("-")
    @test ens.positions == x_pf
    print("-")
    @test ens.weights   == w_pf
    println("DONE")
    
    print("  method dim")
    print("-")
    @test dim(ens) == (d+1) * N
    println("DONE")
    
    print("  method particle_dim")
    print("-")
    @test particle_dim(ens) == d
    println("DONE")
    
    print("  method no_of_particles")
    print("-")
    @test no_of_particles(ens) == N
    println("DONE")
    
    print("  method eff_no_of_particles")
    print("-")
    @test eff_no_of_particles(ens) == 3.3333333333333335
    println("DONE")
    
    print("  method get_pos")
    for i in 1:N
        print("-")
        @test get_pos(ens, i) == x_pf[:, i]
    end
    println("DONE")
    
    print("  method get_weight")
    for i in 1:N
        print("-")
        @test get_weight(ens, i) == w_pf[i]
    end
    println("DONE")
    
    print("  method sum_of_weights")
    print("-")
    @test sum_of_weights(ens) == 1.0
    println("DONE")
    
    print("  method mean")
    print("-")
    @test size(mean(ens)) == (d, 1)
    print("-")
    @test maximum(abs.(mean(ens)[:,1] - [-0.5763499049711248; -0.18267635565605528])) < 1e-15
    println("DONE")
    
    print("  method cov")
    print("-")
    @test size(cov(ens)) == (d, d)
    print("-")
    @test maximum(abs.(cov(ens) - [0.9378181191314968 0.17476858717998095; 0.17476858717998095 0.23760369813722895])) < 1e-15
    println("DONE")
    
    print("  method var")
    print("-")
    @test size(var(ens)) == (d, 1)
    print("-")
    @test maximum(abs.(var(ens)[:,1] - [0.9378181191314968; 0.23760369813722895])) < 1e-15
    println("DONE")
    
    print("  method resample!")
    x_pf_old = deepcopy(x_pf)
    resample!(ens)
    for i in 1:N
        print("-")
        @test get_weight(ens, i) == float(1/N)
    end
    for i in 1:N, j in 1:d
        print("-")
        @test get_pos(ens, i)[j] in x_pf_old
    end
    print("-")
    @test abs(eff_no_of_particles(ens) - N) < 1e-3
    println("DONE")
    
end; #WeightedParticleEnsemble.jl
