using FeedbackParticleFilters, Distributions, Test

println("Testing observation models:")
@testset "Scalar diffusion model" begin
    print("  Scalar diffusion model")
    println("DONE.")
end; #Scalar diffusion model

@testset "Vector diffusion model" begin
    print("  Vector diffusion model")
    println("DONE.")
end; #Vector diffusion model

@testset "Point process model" begin
    print("  Point process model")
    println("DONE.")
end; #Vector diffusion model