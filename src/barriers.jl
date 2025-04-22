export barrier_stop
barrier_stop(nothing, from, delta) = from + delta

struct CardinalBarriers
    start::Float64
    len::Float64
end

cardinals() = (Vec2(1,0), Vec2(0,1), Vec2(-1,0), Vec2(0,-1))

function barrier_stop(b::CardinalBarriers, from, delta)
    # https://stackoverflow.com/questions/563198/whats-the-most-efficent-way-to-calculate-where-two-line-segments-intersect
    shortest_u = 1.0 + 2.0 * eps()
    q = from
    s = delta
    for dir in cardinals()
        p = b.start * dir
        r = b.len * dir
        rxs = cross(r, s)
        if isapprox(rxs, 0.0; atol=1e-8)
            continue
        else
            qmp = q - p
            u = cross(qmp, r) / rxs
            t = cross(qmp, s) / rxs
            if 0.0 <= u < shortest_u && 0.0 <= t <= 1.0
                shortest_u = u
            end
        end
    end
    result = from + (shortest_u - 2.0 * eps()) * delta
    if any(isnan, result)
        @warn "NaN in barrier_stop result. from=$from delta=$delta"
        return from
    end
    return result
end
