using Test
using StaticArrays
using VDPTag2
using POMDPs
using POMDPTools
using ParticleFilters
using Random
using MCTS
using LinearAlgebra

const Vec8 = SVector{8, Float64}
const IVec8 = SVector{8, Int}
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
@testset "ToNextML Action on ParticleCollection" begin
    pomdp = VDPTagPOMDP()
    policy = ToNextML(pomdp; rng=rng)
    particles = ParticleCollection([TagState([0.0, 0.0], [1.0, 1.0]) for _ in 1:20])

    a = action(policy, particles)
    @test isa(a, TagAction)
    @test a.look == false
    @test isfinite(a.angle)
end
@testset "ToNextMLSolver - Continuous Problem" begin
    solver = ToNextMLSolver(rng)
    problem = VDPTagPOMDP()
    policy = solve(solver, problem)

    @test isa(policy, ToNextML)
end
@testset "ToNextMLSolver - Discrete Problem" begin
    solver = ToNextMLSolver(rng)
    problem = AODiscreteVDPTagPOMDP()
    policy = solve(solver, problem)

    @test isa(policy, TranslatedPolicy)
end
@testset "ManageUncertainty Action" begin
    pomdp = VDPTagPOMDP()
    policy = ManageUncertainty(pomdp, 0.1)

    belief = ParticleCollection([
        TagState([0.0, 0.0], [randn(), randn()]) for _ in 1:50
    ])

    a = action(policy, belief)
    @test isa(a, TagAction)
    @test typeof(a.look) == Bool
    @test isfinite(a.angle)
end
@testset "Discretized Interface" begin
    rng = MersenneTwister(123)
    p_adisc = ADiscreteVDPTagPOMDP()
    p_aodisc = AODiscreteVDPTagPOMDP()

    # 1. cproblem
    @test cproblem(p_adisc) isa VDPTagPOMDP

    # 2. convert_o fallback
    x = Vec8(randn(8))
    @test VDPTag2.convert_o(Vec8, x, p_adisc) == x

    # 3. convert_s roundtrip
    state = TagState(Vec2(0.2, -0.4), Vec2(-0.8, 0.9))
    s_idx = VDPTag2.convert_s(Int, state, p_adisc)
    restored = convert_s(TagState, s_idx, p_adisc)
    @test all(isfinite, restored.agent)
    @test all(isfinite, restored.target)

    # 4. convert_a Float64 -> Int
    for θ in range(0.0, stop=2π, length=20)
        i = convert_a(Int, θ, p_adisc)
        @test 1 ≤ i ≤ p_adisc.n_angles
    end

    # 5. convert_a TagAction -> Int
    for look in (true, false), θ in range(0.0, stop=2π, length=5)
        a = TagAction(look, θ)
        i = convert_a(Int, a, p_adisc)
        @test look ? (i > p_adisc.n_angles) : (i ≤ p_adisc.n_angles)
    end

    # 6. n_states
    @test n_states(p_aodisc) == Inf

    # 7. gen + observation (ADiscrete)
    s = TagState(Vec2(0.0, 0.0), Vec2(0.5, 0.5))
    a = convert_a(Int, π/2, p_adisc)
    result = gen(p_adisc, s, a, rng)
    @test result isa NamedTuple
    @test haskey(result, :sp)
    @test haskey(result, :o)
    @test haskey(result, :r)
    _ = observation(p_adisc, s, a, result.sp)

    # 8. gen + observation (AODiscrete)
    s2 = TagState(Vec2(0.0, 0.0), Vec2(-0.8, 0.8))
    a2 = convert_a(Int, π, p_aodisc)
    result2 = gen(p_aodisc, s2, a2, rng)
    @test result2.o isa IVec8
    _ = observation(p_aodisc, s2, a2, result2.sp)
end

