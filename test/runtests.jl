using Test
using VDPTag2
using POMDPs
using POMDPTools
using ParticleFilters
using Random
using MCTS

# Setup
Random.seed!(1)
rng = MersenneTwister(31)
pomdp = VDPTagPOMDP()
mdp_model = mdp(pomdp)
gen = NextMLFirst(mdp_model, rng)
state = TagState(Vec2(1.0, 1.0), Vec2(-1.0, -1.0))

# Dummy Node for MCTS testing
struct DummyNode end
MCTS.n_children(::DummyNode) = rand(1:10)

@testset "ToNextML + MCTS" begin
    @test isa(next_action(gen, pomdp, state, DummyNode()), Float64)
    @test isa(next_action(gen, pomdp, initialstate(pomdp), DummyNode()), Int)
end

@testset "Barrier Stop Sanity" begin
    barriers = CardinalBarriers(0.2, 1.8)
    for a in range(0.0, stop=2π, length=100)
        s = TagState(Vec2(0,0), Vec2(1,1))
        delta = mdp_model.agent_speed * mdp_model.step_size * Vec2(cos(a), sin(a))
        moved = barrier_stop(barriers, s.agent, delta)
        @test norm(moved - s.agent) ≤ norm(delta)
    end
end

@testset "Simulation - Continuous" begin
    policy = ToNextML(pomdp)
    updater = BootstrapFilter(pomdp, 100)
    sim_hist = simulate(HistoryRecorder(max_steps=10), pomdp, policy, updater)
    @test length(state_hist(sim_hist)) > 1
end

@testset "Simulation - Discrete" begin
    dpomdp = AODiscreteVDPTagPOMDP(pomdp)
    policy = RandomPolicy(dpomdp)
    sim_hist = simulate(HistoryRecorder(max_steps=10), dpomdp, policy)
    @test length(state_hist(sim_hist)) > 1
end

@testset "Barriers Block Movement" begin
    pomdp_blocked = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.0, 100.0)))
    policy = ToNextML(pomdp_blocked)
    updater = BootstrapFilter(pomdp_blocked, 100)
    for quadrant in [Vec2(1,1), Vec2(-1,1), Vec2(1,-1), Vec2(-1,-1)]
        for _ in 1:20
            s0 = rand(rng, initialstate(pomdp_blocked))
            s0 = TagState(quadrant, s0.target)
            hist = simulate(HistoryRecorder(max_steps=5), pomdp_blocked, policy, updater, s0)
            @test all(all(s.agent .* quadrant .>= 0.0) for s in state_hist(hist))
        end
    end
end

@testset "No Barriers - Agent Crosses Quadrant" begin
    pomdp_clear = VDPTagPOMDP()
    policy = ToNextML(pomdp_clear)
    updater = BootstrapFilter(pomdp_clear, 100)
    crossed = 0
    for quadrant in [Vec2(1,1), Vec2(-1,1), Vec2(1,-1), Vec2(-1,-1)]
        local crosses = 0
        for _ in 1:50
            s0 = rand(rng, initialstate(pomdp_clear))
            s0 = TagState(quadrant, s0.target)
            hist = simulate(HistoryRecorder(max_steps=10), pomdp_clear, policy, updater, s0)
            if any(s.agent .* quadrant .< 0.0 for s in state_hist(hist))
                crosses += 1
            end
        end
        @test crosses > 0
        crossed += crosses
    end
    @test crossed > 0
end
