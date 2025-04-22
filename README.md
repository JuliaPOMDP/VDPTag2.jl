# VDPTag2

[![CI](https://github.com/JuliaPOMDP/VDPTag2.jl/actions/workflows/Ci.yml/badge.svg)](https://github.com/JuliaPOMDP/VDPTag2.jl/actions/workflows/Ci.yml)[![codecov](https://codecov.io/github/JuliaPOMDP/VDPTag2.jl/graph/badge.svg?token=QKvJhXy8cK)](https://codecov.io/github/JuliaPOMDP/VDPTag2.jl)


See [VDPTag2/test](https://github.com/JuliaPOMDP/VDPTag2.jl/tree/master/test) for usage examples. 

## VDPTag2 with POMCPOW.jl

```jl
using POMDPs
using POMCPOW
using POMDPModels
using POMDPSimulators
using POMDPPolicies
using VDPTag2

solver = POMCPOWSolver(criterion=MaxUCB(20.0))
pomdp = VDPTagPOMDP() # from VDPTag2
planner = solve(solver, pomdp)

hr = HistoryRecorder(max_steps=100)
hist = simulate(hr, pomdp, planner)
for (s, b, a, r, sp, o) in hist
    @show s, a, r, sp
end

rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
println("""
    Cumulative Discounted Reward (for 1 simulation)
        Random: $(discounted_reward(rhist))
        POMCPOW: $(discounted_reward(hist))
    """)
```
VDPTag2 solved via [POMCPOW.jl](https://github.com/JuliaPOMDP/POMCPOW.jl).
