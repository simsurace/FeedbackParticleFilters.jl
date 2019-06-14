using FeedbackParticleFilters

println("Testing basic methods:")
@testset "Base.eltype" begin
    print("  Base.eltype()")
    struct intfilter <: AbstractFilterRepresentation{Int} end
    filter = intfilter()
    print(".")
    @test eltype(filter) == Int
    struct floatfilter <: AbstractFilterRepresentation{Float64} end
    filter = floatfilter()
    print(".")
    @test eltype(filter) == Float64
    struct vectfilter <: AbstractFilterRepresentation{Vector{Float64}} end
    filter = vectfilter()
    print(".")
    @test eltype(filter) == Vector{Float64}
    println("DONE.")
end; #Base.eltype()

@testset "Map" begin
    print("  Map()")
    f(x::Float64) = x
    g(x::Float64) = x^2
    h(x::Float64) = x^3;
    print(".")
    @test Map(f, [1., 2., 3.]) == [1., 2., 3.]
    print(".")
    @test Map([f,g,h], 2.) == [2., 4., 8.]
    print(".")
    @test Map((f,g,h), 3.) == [3., 9., 27.]
    print(".")
    @test Map((f,g,h), [2., 3.], output_shape=1) == [[2., 3.], [4., 9.], [8., 27.]]
    print(".")
    @test Map((f,g,h), [2., 3.], output_shape=2) == [[2., 4., 8.], [3., 9., 27.]]
    print(".")
    @test Map((f,g,h), [2., 3.]) == [[2., 4., 8.], [3., 9., 27.]]
    print(".")
    @test size(Map((f,g,h), [2. 3.; 4. 5.], output_shape=2)) == (2,2)
    print(".")
    @test eltype(Map((f,g,h), [2. 3.; 4. 5.], output_shape=2)) == Array{Float64,1}
    print(".")
    @test Map((f,g,h), [2. 3.; 4. 5.], output_shape=2)[1,2] == [3., 9., 27.]
    print(".")
    @test_throws ErrorException Map((f,g,h), [[1.,2.],[3.,4.]], output_shape=1)
    print(".")
    @test Map((f,g,h), [[1.,2.],[3.,4.]], output_shape=2) == [[[1.0, 1.0, 1.0], [2.0, 4.0, 8.0]], [[3.0, 9.0, 27.0], [4.0, 16.0, 64.0]]]
    
    ff(x::Array{Float64,1}) = x[1]*x[2]
    gg(x::Array{Float64,1}) = x[1]+x[2]
    hh(x::Array{Float64,1}) = x[1]-x[2];
    print(".")
    @test Map((ff,gg,hh), [[1.,2.],[3.,4.]], output_shape=1) == [[2.0, 12.0], [3.0, 7.0], [-1.0, -1.0]]
    print(".")
    @test Map((ff,gg,hh), [[1.,2.],[3.,4.]], output_shape=2) == [[2.0, 3.0, -1.0], [12.0, 7.0, -1.0]]
    println("DONE.")
end; #Map()

f(x::Float64) = x
g(x::Float64) = x^2
h(x::Float64) = x^3;

