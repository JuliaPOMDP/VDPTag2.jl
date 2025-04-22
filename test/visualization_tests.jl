using Test
using Plots
using POMDPTools
using VDPTag2: VDPTagProblem, VDPTagPOMDP, TagState, ParticleCollection, mdp, Vec2

@testset "Visualization Recipes" begin
    # Define pomdp and mdp with visible barriers
    pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.0, 2.0)))
    p = mdp(pomdp)

    @test begin
        plt = plot(p)
        !isempty(plt.series_list)
    end

    # Simulate with updater
    sim = RolloutSimulator()
    updater = BootstrapFilter(pomdp, 10)
    hist = simulate(sim, pomdp, RandomPolicy(pomdp), updater, 3)

    @test begin
        plt = plot(pomdp, hist)
        !isempty(plt.series_list)
    end

    @test begin
        plt = plot(p, hist)
        !isempty(plt.series_list)
    end

    @test begin
        pc = ParticleCollection([TagState([1.0, 2.0], [3.0, 4.0]) for _ in 1:10])
        plt = plot(pc)
        !isempty(plt.series_list)
    end
end
