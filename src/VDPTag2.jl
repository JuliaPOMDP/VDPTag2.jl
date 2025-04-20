module VDPTag2

using POMDPs
using StaticArrays
using Parameters
using Plots
using Distributions
using POMDPTools
using ParticleFilters
using Random
using LinearAlgebra

const Vec2 = SVector{2, Float64}
const Vec8 = SVector{8, Float64}

import Base.rand(rng::AbstractRNG, x::TagState)
import MCTS: next_action, n_children
import ParticleFilters: obs_weight
import POMDPs: actions, isterminal

export
    TagState,
    TagAction,
    VDPTagMDP,
    VDPTagPOMDP,
    Vec2,

    DiscreteVDPTagMDP,
    DiscreteVDPTagPOMDP,
    AODiscreteVDPTagPOMDP,
    ADiscreteVDPTagPOMDP,
    TranslatedPolicy,
    translate_policy,
    cproblem,

    convert_s,
    convert_a,
    convert_o,
    obs_weight,

    ToNextML,
    ToNextMLSolver,
    NextMLFirst,
    DiscretizedPolicy,
    ManageUncertainty,
    CardinalBarriers,
    mdp,
    isterminal

# -------------------------------
# Basic State and Action
# -------------------------------

struct TagState
    agent::Vec2
    target::Vec2
end

struct TagAction
    look::Bool
    angle::Float64
end

# -------------------------------
# MDP and POMDP Definitions
# -------------------------------

@with_kw struct VDPTagMDP{B} <: MDP{TagState, Float64}
    mu::Float64          = 2.0
    agent_speed::Float64 = 1.0
    dt::Float64          = 0.1
    step_size::Float64   = 0.5
    tag_radius::Float64  = 0.1
    tag_reward::Float64  = 100.0
    step_cost::Float64   = 1.0
    pos_std::Float64     = 0.05
    barriers::B          = nothing
    tag_terminate::Bool  = true
    discount::Float64    = 0.95
    goal::Vec2           = Vec2(5.0, 5.0)  # ✅ NEW FIELD
end

@with_kw struct VDPTagPOMDP{B} <: POMDP{TagState, TagAction, Vec8}
    mdp::VDPTagMDP{B}           = VDPTagMDP()
    meas_cost::Float64          = 5.0
    active_meas_std::Float64    = 0.1
    meas_std::Float64           = 5.0
end

const VDPTagProblem = Union{VDPTagMDP,VDPTagPOMDP}
mdp(p::VDPTagMDP) = p
mdp(p::VDPTagPOMDP) = p.mdp

# -------------------------------
# Target Model
# -------------------------------

# -------------------------------
# Target Model
# -------------------------------

function target_speed(p::VDPTagMDP)
    return 0.5  # ✅ you can adjust this constant
end

function next_ml_target(p::VDPTagMDP, target_pos::Vector{Float64})
    goal = p.goal
    direction = goal - target_pos

    if !all(isfinite, target_pos) || !all(isfinite, goal)
        @warn "Non-finite input to next_ml_target: target_pos=$target_pos, goal=$goal"
        return fill(0.0, 2)
    end

    if norm(direction) < 1e-6
        return target_pos
    end

    velocity = normalize(direction) * target_speed(p)
    next_pos = target_pos + velocity * p.dt

    if !all(isfinite, next_pos)
        @warn "next_ml_target returned NaN or Inf: target_pos=$target_pos, velocity=$velocity"
        return target_pos
    end

    return next_pos
end

function next_ml_target(p::VDPTagMDP, target_pos::SVector{2, Float64})
    return next_ml_target(p, collect(target_pos))
end

# -------------------------------
# Transitions and Rewards
# -------------------------------

function POMDPs.transition(pp::VDPTagProblem, s::TagState, a::Float64)
    ImplicitDistribution(pp, s, a) do pp, s, a, rng
        p = mdp(pp)
        targ = next_ml_target(p, s.target) + p.pos_std * SVector(randn(rng), randn(rng))

        if isnan(a)
            error("Generated NaN in transition! Angle 'a' is NaN. State = $s")
        end

        agent_step = p.agent_speed * p.step_size * SVector(cos(a), sin(a))
        agent = barrier_stop(p.barriers, s.agent, agent_step)

        if any(isnan, agent)
            error("Generated NaN in transition! agent=$(agent), target=$(targ), angle=$a")
        end

        return TagState(agent, targ)
    end
end

function POMDPs.reward(pp::VDPTagProblem, s::TagState, a::Float64, sp::TagState)
    p = mdp(pp)
    if norm(sp.agent - sp.target) < p.tag_radius
        return p.tag_reward
    else
        return -p.step_cost
    end
end

POMDPs.discount(pp::VDPTagProblem) = mdp(pp).discount
isterminal(pp::VDPTagProblem, s::TagState) = mdp(pp).tag_terminate && norm(s.agent - s.target) < mdp(pp).tag_radius

# -------------------------------
# Action Spaces
# -------------------------------

struct AngleSpace end
rand(rng::AbstractRNG, ::AngleSpace) = 2π * rand(rng)
POMDPs.actions(::VDPTagMDP) = AngleSpace()

POMDPs.transition(p::VDPTagPOMDP, s::TagState, a::TagAction) = transition(p, s, a.angle)

struct POVDPTagActionSpace end
rand(rng::AbstractRNG, ::POVDPTagActionSpace) = TagAction(rand(rng, Bool), 2π * rand(rng))
POMDPs.actions(::VDPTagPOMDP) = POVDPTagActionSpace()

function POMDPs.reward(p::VDPTagPOMDP, s::TagState, a::TagAction, sp::TagState)
    return reward(mdp(p), s, a.angle, sp) - a.look * p.meas_cost
end

# -------------------------------
# Observation Model
# -------------------------------

struct BeamDist
    abeam::Int
    an::Normal{Float64}
    n::Normal{Float64}
end

function rand(rng::AbstractRNG, d::BeamDist)
    o = MVector{8, Float64}(undef)
    for i in 1:length(o)
        if i == d.abeam
            o[i] = rand(rng, d.an)
        else
            o[i] = rand(rng, d.n)
        end
    end
    return SVector(o)
end

function POMDPs.pdf(d::BeamDist, o::Vec8)
    p = 1.0
    for i in 1:length(o)
        if i == d.abeam
            p *= POMDPs.pdf(d.an, o[i])
        else
            p *= POMDPs.pdf(d.n, o[i])
        end
    end
    return p
end

function active_beam(rel_pos::Vec2)
    if any(isnan, rel_pos)
        error("Invalid relative position: contains NaN")
    end
    angle = mod(atan(rel_pos[2], rel_pos[1]), 2π)
    bm = ceil(Int, 8 * angle / (2π))
    return clamp(bm, 1, 8)
end

function POMDPs.observation(p::VDPTagPOMDP, a::TagAction, sp::TagState)
    rel_pos = sp.target - sp.agent
    if any(isnan, rel_pos)
        error("Observation error: sp.target or sp.agent contains NaN")
    end
    dist = norm(rel_pos)
    abeam = active_beam(rel_pos)
    an = a.look ? Normal(dist, p.active_meas_std) : Normal(dist, p.meas_std)
    n = Normal(1.0, p.meas_std)
    return BeamDist(abeam, an, n)
end

POMDPs.observation(p::VDPTagPOMDP, a::Float64, sp::TagState) = observation(p, TagAction(false, a), sp)

# -------------------------------
# Includes
# -------------------------------

include("rk4.jl")
include("barriers.jl")
include("initial.jl")
include("discretized.jl")
include("visualization.jl")
include("heuristics.jl")

function ModelTools.gbmdp_handle_terminal(pomdp::VDPTagPOMDP, updater::Updater, b::ParticleCollection, s, a, rng)
    return ParticleCollection([s])
end

end # module
