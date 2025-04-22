using Test
using VDPTag2
using POMDPs
using POMDPTools
using ParticleFilters
using Random
using MCTS
using LinearAlgebra
using StaticArrays: @SVector

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

@testset "RK4 and VDP Dynamics" begin
    p = VDPTag2.VDPTagMDP(mu=1.0, dt=0.1)
    pos = VDPTag2.Vec2(1.0, 0.0)
    
    # Check vdp_dynamics directly
    dpos = VDPTag2.vdp_dynamics(p.mu, pos)
    @test isa(dpos, VDPTag2.Vec2)
    @test length(dpos) == 2
    @test isfinite.(dpos) |> all

    # Check rk4step output
    new_pos = VDPTag2.rk4step(p, pos)
    @test isa(new_pos, VDPTag2.Vec2)
    @test length(new_pos) == 2
    @test isfinite.(new_pos) |> all
end

@testset "Plot Recipes Execution" begin
    using Plots

    p = VDPTagPOMDP()
    mdp_obj = mdp(p)
    dummy_hist = simulate(HistoryRecorder(max_steps=3), p, RandomPolicy(p), BootstrapFilter(p, 10))
    pc = ParticleCollection([TagState(Vec2(0.0, 0.0), Vec2(1.0, 1.0)) for _ in 1:5])

    # These checks confirm that plotting doesn't error (basic coverage)
    @testset "Single Object Recipes" begin
        plt1 = plot(mdp_obj)
        @test plt1 isa Plots.Plot
    end

    @testset "POMDP + History Plot" begin
        plt2 = plot(p, dummy_hist)
        @test plt2 isa Plots.Plot
    end

    @testset "Problem + History Plot" begin
        plt3 = plot(mdp_obj, dummy_hist)
        @test plt3 isa Plots.Plot
    end

    @testset "ParticleCollection Plot" begin
        plt4 = plot(pc)
        @test plt4 isa Plots.Plot
    end

    @testset "Quiver Plot" begin
        Plots.quiver(p)  # Should render arrows and call plot! on barriers
        # no assert needed unless we wrap in try/catch
    end
    @testset "DiscreteVDPTagProblem Conversions and Gen" begin
    rng = MersenneTwister(123)
    
    # Create instances of both types
    dpomdp = ADiscreteVDPTagPOMDP()
    aopomdp = AODiscreteVDPTagPOMDP()

    # --- State Conversion ---
    s = TagState(Vec2(0.0, 0.0), Vec2(0.5, -0.5))
    idx = VDPTag2.convert_s(Int, s, dpomdp)
    s_back = VDPTag2.convert_s(TagState, idx, dpomdp)
    @test isa(idx, Int)
    @test isa(s_back, TagState)
    @test isfinite.(s_back.agent) |> all
    @test isfinite.(s_back.target) |> all

    # --- Action Conversion ---
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
    @test typeof(back.look) == Bool

    # --- Observation Conversion ---
    obs = @SVector rand(8)
    o_disc = VDPTag2.convert_o(IVec8, obs, aopomdp)
    @test isa(o_disc, IVec8)

    # --- Action Sampling and Gen ---
    a = rand(POMDPs.actions(dpomdp))
    res1 = POMDPs.gen(dpomdp, s, a, rng)
    res2 = POMDPs.gen(aopomdp, s, a, rng)
    @test haskey(res1, :sp)
    @test haskey(res2, :o)

    # --- Observation Sampling ---
    ob1 = POMDPs.observation(dpomdp, s, a, s)
    ob2 = rand(POMDPs.observation(aopomdp, s, a, s), rng)
    @test isa(ob1, Vec8)
    @test isa(ob2, IVec8)

    # --- Other interface tests ---
    @test POMDPs.discount(dpomdp) ≈ 0.95
    @test !POMDPs.isterminal(dpomdp, idx)
    @test POMDPs.initialstate(dpomdp) isa VDPInitDist
    end
end
