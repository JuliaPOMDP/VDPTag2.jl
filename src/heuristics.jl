import POMDPs: action
using ParticleFilters: particle_mean

function action(p::ToNextML, b::WeightedParticleBelief{TagState})
    s = particle_mean(b)
    return action(p, s)
end

struct ToNextML{RNG<:AbstractRNG} <: Policy
    p::VDPTagMDP
    rng::RNG
end

ToNextML(p::VDPTagProblem; rng=Random.GLOBAL_RNG) = ToNextML(mdp(p), rng)

function POMDPs.action(p::ToNextML, s::TagState)
    next = next_ml_target(p.p, s.target)
    diff = next-s.agent
    return atan(diff[2], diff[1])
end

POMDPs.action(p::ToNextML, b::ParticleCollection{TagState}) = TagAction(false, POMDPs.action(p, rand(p.rng, b)))

struct ToNextMLSolver <: Solver
    rng::AbstractRNG
end

POMDPs.solve(s::ToNextMLSolver, p::VDPTagProblem) = ToNextML(mdp(p), s.rng)
function POMDPs.solve(s::ToNextMLSolver, dp::DiscreteVDPTagProblem)
    cp = cproblem(dp)
    return translate_policy(ToNextML(mdp(cp), s.rng), cp, dp, dp)
end


struct ManageUncertainty <: Policy
    p::VDPTagPOMDP
    max_norm_std::Float64
end

function POMDPs.action(p::ManageUncertainty, b::ParticleCollection{TagState})
    agent = first(particles(b)).agent
    target_particles = Array{Float64}(undef, 2, n_particles(b))
    for (i, s) in enumerate(particles(b))
        target_particles[:,i] = s.target
    end
    normal_dist = fit(MvNormal, target_particles)
    angle = POMDPs.action(ToNextML(mdp(p.p)), TagState(agent, mean(normal_dist)))
    return TagAction(sqrt(det(cov(normal_dist))) > p.max_norm_std, angle)
end

mutable struct NextMLFirst{RNG<:AbstractRNG}
    p::VDPTagMDP
    rng::RNG
end

function next_action(gen::NextMLFirst, mdp::Union{POMDP, MDP}, s::TagState, snode)
    if n_children(snode) < 1
        return POMDPs.action(ToNextML(gen.p, gen.rng), s)
    else
        return 2*pi*rand(gen.rng)
    end
end

function next_action(gen::NextMLFirst, pomdp::Union{POMDP, MDP}, b, onode)
    s = rand(gen.rng, b)
    ca = TagAction(false, next_action(gen, pomdp, s, onode))
    return convert_a(actiontype(pomdp), ca, pomdp)
end

struct TranslatedPolicy{P<:Policy, T, ST, AT} <: Policy
    policy::P
    translator::T
    S::Type{ST}
    A::Type{AT}
end

function translate_policy(p::Policy, from::Union{POMDP,MDP}, to::Union{POMDP,MDP}, translator)
    return TranslatedPolicy(p, translator, statetype(from), actiontype(to))
end

function POMDPs.action(p::TranslatedPolicy, s)
    cs = convert_s(p.S, s, p.translator)
    ca = POMDPs.action(p.policy, cs)
    return convert_a(p.A, ca, p.translator)
end

# this is not the most efficient way to do this
function POMDPs.action(p::TranslatedPolicy, pc::AbstractParticleBelief)
    @assert !isa(pc, WeightedParticleBelief)
    cpc = ParticleCollection([convert_s(p.S, s, p.translator) for s in particles(pc)])
    ca = POMDPs.action(p.policy, cpc)
    return convert_a(p.A, ca, p.translator)
end
