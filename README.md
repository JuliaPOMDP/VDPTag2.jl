# VDPTag2

[![Build Status](https://travis-ci.org/zsunberg/VDPTag2.jl.svg?branch=master)](https://travis-ci.org/zsunberg/VDPTag2.jl)[![codecov.io](http://codecov.io/github/zsunberg/VDPTag2.jl/coverage.svg?branch=master)](http://codecov.io/github/zsunberg/VDPTag2.jl?branch=master)


See [VDPTag2/test](https://github.com/zsunberg/VDPTag2.jl/tree/master/test) for usage examples. 

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
