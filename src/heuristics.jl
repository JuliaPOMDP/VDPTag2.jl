import POMDPs: action, solve
import ParticleFilters: ParticleCollection, WeightedParticleBelief, particles, n_particles
using Distributions: MvNormal, fit, mean, cov
using Random

# ----------------------------
# ToNextML Policy
# ----------------------------

struct ToNextML{RNG<:AbstractRNG} <: Policy
    p::VDPTagMDP
    rng::RNG
end

ToNextML(p::VDPTagProblem; rng=Random.GLOBAL_RNG) = ToNextML(mdp(p), rng)

function POMDPs.action(p::ToNextML, s::TagState)
    next = next_ml_target(p.p, s.target)
    diff = next - s.agent

    @assert all(isfinite, s.agent) "Agent state has non-finite values: $(s.agent)"
    @assert all(isfinite, next) "Next target position has non-finite values: $(next)"
    @assert all(isfinite, diff) "Diff vector has non-finite values: $(diff)"

    angle = atan(diff[2], diff[1])
    @assert isfinite(angle) "Computed angle is not finite. diff = $diff"

    return angle
end

function POMDPs.action(p::ToNextML, b::ParticleCollection{TagState})
    s = rand(p.rng, b)
    return TagAction(false, POMDPs.action(p, s))
end

function POMDPs.action(p::ToNextML, b::WeightedParticleBelief{TagState})
    s = particle_mean(b)
    return TagAction(false, POMDPs.action(p, s))
end

# ----------------------------
# Solver
# ----------------------------

struct ToNextMLSolver <: Solver
    rng::AbstractRNG
end

POMDPs.solve(s::ToNextMLSolver, p::VDPTagProblem) = ToNextML(mdp(p), s.rng)

function POMDPs.solve(s::ToNextMLSolver, dp::DiscreteVDPTagProblem)
    cp = cproblem(dp)
    return translate_policy(ToNextML(mdp(cp), s.rng), cp, dp, dp)
end

# ----------------------------
# ManageUncertainty Policy
# ----------------------------
struct ManageUncertainty <: Policy
    p::VDPTagPOMDP
    max_norm_std::Float64
end
function POMDPs.action(p::ManageUncertainty, b::ParticleCollection{TagState})
    agent = first(particles(b)).agent
    target_particles = Matrix(hcat([s.target for s in particles(b)]...))

    # Fit with fallback in case of numerical issues
    try
        normal_dist = fit(MvNormal, target_particles)
    catch e
        if isa(e, PosDefException)
            μ = vec(mean(target_particles, dims=2))  # ensure 1D vector
            Σ = cov(target_particles) + 1e-6I        # regularize
            normal_dist = MvNormal(μ, Σ)
        else
            rethrow(e)
        end
    end

    mean_target = mean(normal_dist)
    uncertainty = sqrt(det(cov(normal_dist)))

    angle = POMDPs.action(ToNextML(mdp(p.p)), TagState(agent, mean_target))
    look = uncertainty > p.max_norm_std

    return TagAction(look, angle)
end

# ----------------------------
# NextMLFirst Heuristic
# ----------------------------

mutable struct NextMLFirst{RNG<:AbstractRNG}
    p::VDPTagMDP
    rng::RNG
end

function next_action(gen::NextMLFirst, mdp::Union{POMDP, MDP}, s::TagState, snode)
    if n_children(snode) < 1
        return POMDPs.action(ToNextML(gen.p, gen.rng), s)
    else
        return 2π * rand(gen.rng)
    end
end

function next_action(gen::NextMLFirst, pomdp::Union{POMDP, MDP}, b, onode)
    s = rand(gen.rng, b)
    ca = TagAction(false, next_action(gen, pomdp, s, onode))
    return convert_a(actiontype(pomdp), ca, pomdp)
end

# ----------------------------
# Policy Translation
# ----------------------------

struct TranslatedPolicy{P<:Policy, T, ST, AT} <: Policy
    policy::P
    translator::T
    S::Type{ST}
    A::Type{AT}
end

function translate_policy(p::Policy, from::Union{POMDP, MDP}, to::Union{POMDP, MDP}, translator)
    return TranslatedPolicy(p, translator, statetype(from), actiontype(to))
end

function POMDPs.action(p::TranslatedPolicy, s)
    cs = convert_s(p.S, s, p.translator)
    ca = POMDPs.action(p.policy, cs)
    return convert_a(p.A, ca, p.translator)
end

function POMDPs.action(p::TranslatedPolicy, pc::AbstractParticleBelief)
    @assert !isa(pc, WeightedParticleBelief)
    cpc = ParticleCollection([convert_s(p.S, s, p.translator) for s in particles(pc)])
    ca = POMDPs.action(p.policy, cpc)
    return convert_a(p.A, ca, p.translator)
end

# ----------------------------
# Particle Mean Utility
# ----------------------------

function particle_mean(b::WeightedParticleBelief{TagState})
    ps = particles(b)
    ws = weights(b)

    agent = ps[1].agent  # assume consistent across particles
    target = sum(w * p.target for (w, p) in zip(ws, ps))
    return TagState(agent, target)
end
