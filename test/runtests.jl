using Test
using VDPTag2
using POMDPs
using POMDPTools
using ParticleFilters
using Random
using MCTS
using LinearAlgebra

# Seed RNG for reproducibility
Random.seed!(1)
rng = MersenneTwister(31)

@testset "ToNextML + MCTS" begin
    pomdp = VDPTagPOMDP()
    gen = NextMLFirst(mdp(pomdp), rng)
    s = TagState(Vec2(1.0, 1.0), Vec2(-1.0, -1.0))

    struct DummyNode end
    MCTS.n_children(::DummyNode) = rand(1:10)

    a1 = next_action(gen, pomdp, s, DummyNode())
    a2 = next_action(gen, pomdp, initialstate(pomdp), DummyNode())

    @test a1 isa Float64
    @test a2 isa TagAction
    @test a2.look == false
    @test 0.0 <= a2.angle <= 2π
end

@testset "Barrier Stop Sanity" begin
    barriers = CardinalBarriers(0.2, 1.8)
    for a in range(0.0, stop=2π, length=100)
        s = TagState(Vec2(0, 0), Vec2(1, 1))
        delta = 1.0 * 0.5 * Vec2(cos(a), sin(a))  # speed * step_size
        moved = barrier_stop(barriers, s.agent, delta)
        @test norm(moved - s.agent) ≤ norm(delta) + 1e-8
    end
end

@testset "Simulation - Continuous" begin
    pomdp = VDPTagPOMDP()
    policy = ToNextML(pomdp)
    updater = BootstrapFilter(pomdp, 100)
    hist = simulate(HistoryRecorder(max_steps=10), pomdp, policy, updater)
    @test length(state_hist(hist)) > 1
end

@testset "Simulation - Discrete" begin
    dpomdp = AODiscreteVDPTagPOMDP()
    policy = RandomPolicy(dpomdp)
    hist = simulate(HistoryRecorder(max_steps=10), dpomdp, policy)
    @test length(state_hist(hist)) > 1
end

function sample_in_quadrant(rng, quadrant)
    agent = rand(rng, Vec2) .* 5.0 .* quadrant
    target = rand(rng, Vec2) .* 5.0 .* quadrant
    return TagState(agent, target)
end

@testset "Barriers Block Movement" begin
    pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.0, 100.0)))
    policy = ToNextML(pomdp)
    updater = BootstrapFilter(pomdp, 100)

    for quadrant in [Vec2(1, 1), Vec2(-1, 1), Vec2(1, -1), Vec2(-1, -1)]
        for _ in 1:10
            s0 = sample_in_quadrant(rng, quadrant)
            hist = simulate(HistoryRecorder(max_steps=5), pomdp, policy, updater, s0)
            violations = count(s -> any(s.agent .* quadrant .< -1e-6), state_hist(hist))
            @test violations ≤ 4
        end
    end
end

@testset "No Barriers - Can Cross Quadrants" begin
    pomdp = VDPTagPOMDP()
    policy = ToNextML(pomdp)
    updater = BootstrapFilter(pomdp, 100)

    for quadrant in [Vec2(1, 1), Vec2(-1, 1), Vec2(1, -1), Vec2(-1, -1)]
        crossed = 0
        for _ in 1:25
            s0 = sample_in_quadrant(rng, quadrant)
            hist = simulate(HistoryRecorder(max_steps=10), pomdp, policy, updater, s0)
            if any(any(s.agent .* quadrant .< 0.0) for s in state_hist(hist))
                crossed += 1
            end
        end
        @test crossed > 0  # should cross at least once
    end
end

@testset "Heuristics and Discretized Coverage" begin
    # Heuristics
    pomdp = VDPTagPOMDP()
    mdp_model = mdp(pomdp)
    policy = ToNextML(mdp_model, rng)
    trans = translate_policy(policy, mdp_model, ADiscreteVDPTagPOMDP(), ADiscreteVDPTagPOMDP())
    s = TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0))
    act = POMDPs.action(trans, s)
    @test act isa Int

    belief_states = [TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0 + 0.01*i)) for i in 1:10]
    belief = ParticleCollection(belief_states)
    policy_uncert = ManageUncertainty(pomdp, 0.001)
    a = POMDPs.action(policy_uncert, belief)
    @test a isa TagAction

    # Discretized conversions
    dp = AODiscreteVDPTagPOMDP()
    s_int = convert_s(Int, s, dp)
    @test s_int isa Int
    s_back = convert_s(TagState, s_int, dp)
    @test s_back isa TagState

    tagact = TagAction(true, π/2)
    a_int = convert_a(Int, tagact, dp)
    @test a_int isa Int
    a_back = convert_a(TagAction, a_int, dp)
    @test a_back isa TagAction

    obs_vec = rand(rng, observation(cproblem(dp), s, tagact, s))
    obs_disc = convert_o(IVec8, obs_vec, dp)
    @test obs_disc isa IVec8

    # Gen and observation
    gen1 = POMDPs.gen(dp, s, a_int, rng)
    @test gen1.sp isa TagState

    obs = rand(rng, POMDPs.observation(dp, s, a_int, s))
    @test obs isa IVec8
end
