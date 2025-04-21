@recipe function f(p::VDPTagProblem)
    m = mdp(p)
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    bs = m.barriers
    # @series begin
    #     lim = (-3.5,3.5)
    #     pts = linspace( -3, 3, 20)
    #     xys = [Vec2(x, y) for x in pts, y in pts]
    #     xs = [xy[1] for xy in xys]
    #     ys = [xy[2] for xy in xys]
    #     seriestype := quiver
    #     label := ""
    #     color --> :lightblue
    #     quiver := (x,y)->0.1*vdp_dynamics(m.mu, Vec2(x,y)),
    #     xs, ys
    # end
    if bs isa CardinalBarriers
        for dir in cardinals()
            ends = (bs.start*dir, (bs.start+bs.len)*dir)
            color := :black
            linewidth --> 4
            label --> ""
            @series [v[1] for v in ends], [v[2] for v in ends]
        end
    end
end

@recipe function f(pomdp::VDPTagPOMDP, h::POMDPTools.AbstractSimHistory)
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    @series mdp(pomdp), h
    @series begin
        label := "belief"
        VDPTag2.ParticleCollection(particles(belief_hist(h)[end]))
    end
end


end

@recipe function f(p::VDPTagProblem, h::POMDPTools.AbstractSimHistory)
    m = mdp(p)
    ratio --> :equal
    xlim --> (-5, 5)
    ylim --> (-5, 5)
    @series begin
        label := "path"
        x = [s.agent[1] for s in state_hist(h)[1:end-1]]
        y = [s.agent[2] for s in state_hist(h)[1:end-1]]
        x, y
    end
    @series begin
        a = h[end].a
        if a isa TagAction && a.look
            color := :blue
        else
            color := :red
        end
        s = state_hist(h)[end-1]
        label := "current agent position"
        pts = Plots.partialcircle(0, 2*pi, 100, m.tag_radius)
        x, y = Plots.unzip(pts)
        x.+s.agent[1], y.+s.agent[2]
    end
    @series begin
        seriestype := :scatter
        label := "current target"
        pos = state_hist(h)[end-1].target
        color --> :orange
        [pos[1]], [pos[2]]
    end
    @series begin m end
end


@recipe function f(pc::ParticleCollection{TagState})
    seriestype := :scatter
    x = [p.target[1] for p in particles(pc)]
    y = [p.target[2] for p in particles(pc)]
    color --> :black
    markersize --> 0.1
    x, y
end

"Create a quiver plot of the equations and the barriers"
function Plots.quiver(p::VDPTagProblem)
    m = mdp(p)
    lim = (-3.5,3.5)
    pts = range( -3, stop=3, length=16)
    xys = [Vec2(x, y) for x in pts, y in pts]
    xs = [xy[1] for xy in xys]
    ys = [xy[2] for xy in xys]
    quiver(xs, ys,
            quiver = (x,y)->0.1*vdp_dynamics(m.mu, Vec2(x,y)),
            color=:lightblue,
          )
    plot!(p) # to get barriers
    plot!(xlim=lim, ylim=lim)
end
