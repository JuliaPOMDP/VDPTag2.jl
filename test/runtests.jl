import POMDPs: sampletype
using Test
using VDPTag2
using POMDPs
using POMDPTools
using ParticleFilters
using Random
using MCTS
using LinearAlgebra
using StaticArrays: @SVector
using Plots

# Type aliases for easier usage
const IVec8 = VDPTag2.IVec8
const Vec8 = VDPTag2.Vec8

# Import key types from VDPTag2
import VDPTag2: VDPInitDist, VDPTagProblem, VDPTagMDP, CardinalBarriers

# Set RNG seed for reproducibility
Random.seed!(1)
rng = MersenneTwister(31)

# ----------------------------
# Policy and Action Logic Tests
# ----------------------------

@testset "ToNextML + MCTS" begin
    # Test deterministic and belief-based action selection
    pomdp = VDPTagPOMDP()
    gen = NextMLFirst(mdp(pomdp), rng)
    s = TagState(Vec2(1.0, 1.0), Vec2(-1.0, -1.0))

    struct DummyNode end
    MCTS.n_children(::DummyNode) = rand(1:10)

    a1 = next_action(gen, pomdp, s, DummyNode())
    a2 = next_action(gen, pomdp, initialstate(pomdp), DummyNode())

    @test isa(a1, Float64)
    @test isa(a2, TagAction)
    @test !a2.look
    @test 0.0 ≤ a2.angle ≤ 2π
end

@testset "Barrier Stop Sanity" begin
    # Ensure barrier logic correctly blocks movement
    barriers = CardinalBarriers(0.2, 1.8)
    for a in range(0.0, stop=2π, length=100)
        s = TagState(Vec2(0, 0), Vec2(1, 1))
        delta = 0.5 * Vec2(cos(a), sin(a))
        moved = barrier_stop(barriers, s.agent, delta)
        @test norm(moved - s.agent) ≤ norm(delta) + 1e-8
    end
end

@testset "Simulation - Continuous" begin
    # Validate simulation using continuous model
    pomdp = VDPTagPOMDP()
    policy = ToNextML(pomdp)
    updater = BootstrapFilter(pomdp, 100)
    hist = simulate(HistoryRecorder(max_steps=10), pomdp, policy, updater)
    @test length(state_hist(hist)) > 1
end

@testset "Simulation - Discrete" begin
    # Validate simulation using discrete model
    dpomdp = AODiscreteVDPTagPOMDP()
    policy = RandomPolicy(dpomdp)
    hist = simulate(HistoryRecorder(max_steps=10), dpomdp, policy)
    @test length(state_hist(hist)) > 1
end

# Helper for generating test states in different quadrants
function sample_in_quadrant(rng, quadrant)
    TagState(rand(rng, Vec2) .* 5.0 .* quadrant, rand(rng, Vec2) .* 5.0 .* quadrant)
end

@testset "Barriers Block Movement" begin
    # Ensure agents are blocked by barriers in all directions
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
    # Confirm agent can cross between quadrants without barriers
    pomdp = VDPTagPOMDP()
    policy = ToNextML(pomdp)
    updater = BootstrapFilter(pomdp, 100)

    for quadrant in [Vec2(1, 1), Vec2(-1, 1), Vec2(1, -1), Vec2(-1, -1)]
        crossed = 0
        for _ in 1:25
            s0 = sample_in_quadrant(rng, quadrant)
            hist = simulate(HistoryRecorder(max_steps=10), pomdp, policy, updater, s0)
            crossed += any(s -> any(s.agent .* quadrant .< 0.0), state_hist(hist)) ? 1 : 0
        end
        @test crossed > 0
    end
end

@testset "ToNextML Action on ParticleCollection" begin
    # Check policy behavior on unweighted particle belief
    pomdp = VDPTagPOMDP()
    policy = ToNextML(pomdp; rng=rng)
    particles = ParticleCollection([TagState([0.0, 0.0], [1.0, 1.0]) for _ in 1:20])
    a = action(policy, particles)

    @test isa(a, TagAction)
    @test !a.look
    @test isfinite(a.angle)
end

@testset "ToNextMLSolver - Continuous Problem" begin
    # Solver should return correct policy type
    solver = ToNextMLSolver(rng)
    policy = solve(solver, VDPTagPOMDP())
    @test isa(policy, ToNextML)
end

@testset "ToNextMLSolver - Discrete Problem" begin
    # Solver for discrete problem returns TranslatedPolicy
    solver = ToNextMLSolver(rng)
    policy = solve(solver, AODiscreteVDPTagPOMDP())
    @test isa(policy, TranslatedPolicy)
end

@testset "ManageUncertainty Action" begin
    # Policy should trigger 'look' when uncertainty is high
    pomdp = VDPTagPOMDP()
    policy = ManageUncertainty(pomdp, 0.1)
    belief = ParticleCollection([TagState([0.0, 0.0], [randn(), randn()]) for _ in 1:50])
    a = action(policy, belief)

    @test isa(a, TagAction)
    @test isa(a.look, Bool)
    @test isfinite(a.angle)
end

@testset "RK4 and VDP Dynamics" begin
    # Test numerical integration and derivative function
    p = VDPTag2.VDPTagMDP(mu=1.0, dt=0.1)
    pos = VDPTag2.Vec2(1.0, 0.0)

    dpos = VDPTag2.vdp_dynamics(p.mu, pos)
    @test isa(dpos, VDPTag2.Vec2)
    @test length(dpos) == 2
    @test all(isfinite, dpos)

    new_pos = VDPTag2.rk4step(p, pos)
    @test isa(new_pos, VDPTag2.Vec2)
    @test length(new_pos) == 2
    @test all(isfinite, new_pos)
end

@testset "Plot Recipes Execution" begin
    # Ensure custom plot recipes execute without error
    p = VDPTagPOMDP()
    mdp_obj = mdp(p)
    dummy_hist = simulate(HistoryRecorder(max_steps=3), p, RandomPolicy(p), BootstrapFilter(p, 10))
    pc = ParticleCollection([TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0)) for _ in 1:5])

    @test plot(mdp_obj) isa Plots.Plot
    @test plot(p, dummy_hist) isa Plots.Plot
    @test plot(mdp_obj, dummy_hist) isa Plots.Plot
    @test plot(pc) isa Plots.Plot
    Plots.quiver(p)
end

@testset "DiscreteVDPTagProblem Conversions and Gen" begin
    # Test conversion functions for discrete problem
    dpomdp = ADiscreteVDPTagPOMDP()
    aopomdp = AODiscreteVDPTagPOMDP()
    rng = MersenneTwister(123)

    s = TagState(Vec2(0.0, 0.0), Vec2(0.5, -0.5))
    idx = VDPTag2.convert_s(Int, s, dpomdp)
    s_back = VDPTag2.convert_s(TagState, idx, dpomdp)
    @test isa(idx, Int)
    @test isa(s_back, TagState)

    for angle in [0.0, π/2, π, 3π/2]
        i = VDPTag2.convert_a(Int, angle, dpomdp)
        a = VDPTag2.convert_a(Float64, i, dpomdp)
        @test isa(i, Int)
        @test isfinite(a)
    end

    tag_action = TagAction(true, π/4)
    ai = VDPTag2.convert_a(Int, tag_action, dpomdp)
    back = VDPTag2.convert_a(TagAction, ai, dpomdp)
    @test isa(back, TagAction)

    obs = @SVector rand(8)
    o_disc = VDPTag2.convert_o(IVec8, obs, aopomdp)
    @test isa(o_disc, IVec8)

    a = rand(POMDPs.actions(dpomdp))
    res1 = POMDPs.gen(dpomdp, s, a, rng)
    res2 = POMDPs.gen(aopomdp, s, a, rng)
    @test haskey(res1, :sp)
    @test haskey(res2, :o)

    ob1 = POMDPs.observation(dpomdp, s, a, s)
    ob2 = rand(rng, POMDPs.observation(aopomdp, s, a, s))
    @test isa(rand(rng, ob1), Vec8)
    @test isa(ob2, IVec8)

    @test isapprox(POMDPs.discount(dpomdp), 0.95)
    @test !POMDPs.isterminal(dpomdp, idx)
    @test POMDPs.initialstate(dpomdp) isa VDPInitDist
end

@testset "VDPInitDist Sampletype Coverage" begin
    # Covers sampletype(::Type{VDPInitDist})
    @test VDPTag2.sampletype(VDPTag2.VDPInitDist) == VDPTag2.TagState
end

@testset "Plot VDPTagMDP with CardinalBarriers" begin
    # Covers plotting logic for CardinalBarriers
    m = VDPTagMDP(barriers=CardinalBarriers(0.2, 1.8))
    plt = plot(m)
    @test plt isa Plots.Plot
end

@testset "Plot ParticleCollection{TagState}" begin
    # Covers plot recipe for unweighted belief visualization
    pcs = [TagState([0.0, 0.0], [randn(), randn()]) for _ in 1:10]
    pc = ParticleCollection(pcs)
    plt = plot(pc)
    @test plt isa Plots.Plot
end

@testset "TranslatedPolicy Action Coverage" begin
    # Covers action translation logic for discretized POMDP
    dpomdp = ADiscreteVDPTagPOMDP()
    rng = MersenneTwister(123)
    cp = cproblem(dpomdp)
    policy = ToNextML(mdp(cp), rng)
    translated = translate_policy(policy, cp, dpomdp, dpomdp)

    s = TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0))
    a = POMDPs.action(translated, s)
    @test isa(a, Int)

    pc = ParticleCollection([TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0)) for _ in 1:5])
    a2 = POMDPs.action(translated, pc)
    @test isa(a2, Int)
end

@testset "NextMLFirst Coverage" begin
    # Cover fallback case where no children exist
    mdp_obj = VDPTagMDP()
    gen = NextMLFirst(mdp_obj, rng)
    s = TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0))

    struct DummyNodeZero end
    VDPTag2.n_children(::DummyNodeZero) = 0

    a = next_action(gen, mdp_obj, s, DummyNodeZero())
    @test isa(a, Float64)
end
