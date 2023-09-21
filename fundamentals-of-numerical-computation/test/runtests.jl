using MyFNC
using Test
using LinearAlgebra


@testset "backsub" begin
    U = [1 2 3; 0 4 5; 0 0 6]
    b = [1, 2, 3]
    @test MyFNC.backsub(U, b) ≈ U\b

    U = [0 0; 0 0]
    b = [1, 2]
    @test_throws SingularException U\b
    @test_throws SingularException MyFNC.backsub(U, b)
end;


@testset "forwardsub" begin
    L = [1 0 0; 2 3 0; 4 5 6]
    b = [1, 2, 3]
    @test MyFNC.forwardsub(L, b) ≈ L\b

    L = [0 0; 0 0]
    b = [1, 2]
    @test_throws SingularException L\b
    @test_throws SingularException MyFNC.forwardsub(L, b)
end;