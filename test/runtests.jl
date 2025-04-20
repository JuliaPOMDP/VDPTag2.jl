using Test
using VDPTag2
using POMDPs
using Random
using ParticleFilters
using Distributions

# Use a consistent random seed
rng = MersenneTwister(123)

@testset "Discrete State, Action, and Observation Conversion" begin
    dp = AODiscreteVDPTagPOMDP()

    # Create a simple state and action
    s = TagState(Vec2(0.2, 0.3), Vec2(-0.5, -0.4))
    a = TagAction(true, π/4)

    # Test state conversion to integer and back
    s_int = convert_s(Int, s, dp)
    s_back = convert_s(TagState, s_int, dp)
    @test s_back isa TagState

    # Test action conversion to integer and back
    a_int = convert_a(Int, a, dp)
    a_back = convert_a(TagAction, a_int, dp)
    @test a_back isa TagAction

    # Test observation conversion
    obs_vec = rand(rng, observation(cproblem(dp), s, a, s))
    obs_disc = convert_o(IVec8, obs_vec, dp)
    @test obs_disc isa IVec8
end

@testset "Discrete Gen and Observation Functions" begin
    dp1 = AODiscreteVDPTagPOMDP()
    dp2 = ADiscreteVDPTagPOMDP()
    s = TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0))
    a = 1  # integer action

    # Run gen function and check types
    r1 = POMDPs.gen(dp1, s, a, rng)
    r2 = POMDPs.gen(dp2, s, a, rng)
    @test r1.sp isa TagState
    @test r2.sp isa TagState

    # Run observation function
    o1 = rand(rng, POMDPs.observation(dp1, s, a, s))
    o2 = POMDPs.observation(dp2, s, a, s)
    @test o1 isa IVec8
    @test o2 isa Vec8
end

@testset "Heuristic Policies and Translations" begin
    pomdp = VDPTagPOMDP()
    mdp_model = mdp(pomdp)

    # ToNextML policy chooses direction based on current state
    policy = ToNextML(mdp_model; rng=rng)
    s = TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0))
    angle = POMDPs.action(policy, s)
    @test angle isa Float64

    # ManageUncertainty policy adds 'look' based on uncertainty
    states = [TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0 + 0.01*i)) for i in 1:50]
    belief = ParticleCollection(states)
    policy2 = ManageUncertainty(pomdp, 0.01)
    a2 = POMDPs.action(policy2, belief)
    @test a2 isa TagAction

    # TranslatedPolicy maps actions from one model space to another
    dp = ADiscreteVDPTagPOMDP()
    translated = translate_policy(policy, mdp_model, dp, dp)
    a_translated = POMDPs.action(translated, s)
    @test a_translated isa Int
end
