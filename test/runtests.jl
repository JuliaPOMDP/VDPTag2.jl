using Test
using VDPTag2
using POMDPs
using POMDPTools
using ParticleFilters
using Random
using Distributions

@testset "ToNextMLSolver.solve and TranslatedPolicy.action" begin
    dpomdp = ADiscreteVDPTagPOMDP()
    solver = ToNextMLSolver(MersenneTwister(123))
    policy = solve(solver, dpomdp)

    # Test action on discrete state
    s = convert_s(Int, TagState(Vec2(0.0, 0.0), Vec2(0.0, 0.0)), dpomdp)
    a = action(policy, s)
    @test a isa Int
end

@testset "ManageUncertainty.action on ParticleCollection" begin
    pomdp = VDPTagPOMDP()
    policy = ManageUncertainty(pomdp, 0.01)

    particles_ = [TagState(Vec2(0.0, 0.0), Vec2(x, x)) for x in 0.0:0.1:0.9]
    b = ParticleCollection(particles_)

    a = action(policy, b)
    @test a isa TagAction
    @test a.look == true
end

@testset "TranslatedPolicy.action on ParticleCollection" begin
    dpomdp = ADiscreteVDPTagPOMDP()
    solver = ToNextMLSolver(MersenneTwister(123))
    base_policy = ToNextML(mdp(dpomdp.cpomdp), solver.rng)
    translated = translate_policy(base_policy, dpomdp.cpomdp, dpomdp, dpomdp)

    # Construct belief and test
    s = TagState(Vec2(0.0, 0.0), Vec2(0.0, 0.0))
    b = ParticleCollection([s for _ in 1:10])
    a = action(translated, b)
    @test a isa Int
end

@testset "particle_mean utility" begin
    particles_ = [TagState(Vec2(0.0, 0.0), Vec2(x, x)) for x in 0.0:0.1:0.9]
    weights_ = fill(1/length(particles_), length(particles_))
    b = WeightedParticleBelief(particles_, weights_)
    s_mean = particle_mean(b)
    @test s_mean isa TagState
    @test isapprox(s_mean.target[1], 0.4, atol=0.1)
end
