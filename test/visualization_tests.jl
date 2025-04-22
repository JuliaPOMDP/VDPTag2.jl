using Test
using Plots
using POMDPTools
using VDPTag2: VDPTagProblem, VDPTagPOMDP, TagState, ParticleCollection, mdp, Vec2

@testset "Visualization Recipes" begin
    # Create a dummy VDPTagProblem
    m = VDPTagMDP()
    p = VDPTagProblem(m)

    # Plot the VDPTagProblem (covers: @recipe function f(p::VDPTagProblem))
    @test begin
        plt = plot(p)
        !isempty(plt.series_list)
    end

    # Create a dummy history
    pomdp = VDPTagPOMDP(m)
    sim = RolloutSimulator()
    hist = simulate(sim, pomdp, RandomPolicy(pomdp), 3)

    # Plot with POMDP and history (covers: @recipe function f(pomdp::VDPTagPOMDP, h::...))
    @test begin
        plt = plot(pomdp, hist)
        !isempty(plt.series_list)
    end

    # Plot with problem and history (covers: @recipe function f(p::VDPTagProblem, h::...))
    @test begin
        plt = plot(p, hist)
        !isempty(plt.series_list)
    end

    # Plot particle collection (covers: @recipe function f(pc::ParticleCollection{TagState}))
    @test begin
        pc = ParticleCollection([TagState([1.0, 2.0], [3.0, 4.0]) for _ in 1:10])
        plt = plot(pc)
        !isempty(plt.series_list)
    end
end
