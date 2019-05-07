function getlaydown(h, g::AirDefenseGame, C, A)
    len = size(g.X, 1)
    cities = DataFrame(id = 1:len, x = g.X[:,1], y = g.X[:,2], value = g.v,
        full_coverage = fill(0, len), partial_coverage = fill(0, len),
        iads = fill(false, len), physical_attack = fill(false, len),
        cyber_defense = fill(false, len), cyber_attack = fill(false, len),
        iads_status = fill("none", len))
    for i in g.iads # physical defenses
        cities.iads[i] = true
    end
    for i in A[6][h[6]] # physically attacked cities
        cities.physical_attack[i] = true
    end
    for i in getindex(g.iads, getindex(A[1][h[1]], A[5][h[5]])) # cyber-defended iads
        cities.cyber_defense[i] = true
    end
    for i in getindex(g.iads, getindex(A[2][h[2]], A[3][h[3]])) # cyber-attacked iads
        cities.cyber_attack[i] = true
    end
    for i in 1:len
        if cities.iads[i] == true && cities.cyber_attack[i] == false
            cities.iads_status[i] = "full"
        elseif cities.iads[i] == true && cities.cyber_attack[i] == true && cities.cyber_defense[i] == true
            cities.iads_status[i] = "partial"
        elseif cities.iads[i] == true && cities.cyber_attack[i] == true && cities.cyber_defense[i] == false
            cities.iads_status[i] = "inoperable"
        end
    end
    full_defenders = @where(cities, :iads_status .== "full")
    partial_defenders = @where(cities, :iads_status .== "partial")
    for i in 1:len
        for j in 1:size(full_defenders, 1)
            if C[full_defenders.id[j], cities.id[i]] == 1
                cities.full_coverage[i] += 1
            end
        end
        for pd in 1:size(partial_defenders, 1)
            if C[partial_defenders.id[pd], cities.id[i]] == 1
                cities.partial_coverage[i] += 1
            end
        end
    end
    cities
end

function plot_laydown(df::DataFrame)
    clibrary(:Plots)
    df_dc = @where(df, :cyber_defense .== true)
    df_ac = @where(df, :cyber_attack .== true)
    df_fc = @where(df, :iads_status .== "full")
    df_pc = @where(df, :iads_status .== "partial")
    df_nc = @where(df, :iads_status .== "inoperable")
    df_ap = @where(df, :physical_attack .== true)
    # plot cities
    p = plot(df.x, df.y, seriestype = :scatter, markershape = :circle, markercolor = :black,
        markersize = 1, markeralpha = 0.5, legend = :none,
        xlabel = "Longitude (Regularized)", ylabel = "Latitude (Regularized)",
        label = "Cities")
    # plot cyber defenses
    plot!(p, df_dc.x, df_dc.y, seriestype = :scatter, markershape = :circle,
        markeralpha = 0.2, markercolor = :blue, markersize = 10,
        label = "Cyber Defense")
    # plot cyber attacks
    plot!(p, df_ac.x, df_ac.y, seriestype = :scatter, markershape = :+,
        markeralpha = 0.0, markercolor = :purple, markersize = 7,
        markerstrokealpha = 1.0, markerstrokecolor = :purple, markerstrokewidth = 0.0001,
        markerstrokestyle = :dash, label = "Cyber Attack")
    # label cities with value
    for i in 1:size(df, 1)
        annotate!(p, df.x[i], df.y[i], text(string("  ", round(df.value[i], digits = 2)), 10, :left, :darkgray))
        annotate!(p, df.x[i], df.y[i], text(string(df.id[i], ""), 10, :bottom, :black))
    end
    # plot coverage
    for i in 1:size(df, 1)
        if df.iads[i] == true
            cx, cy = circlecoord(df.x[i], df.y[i], g.r)
            style = df.iads_status[i] == "full" ? :solid : df.iads_status[i] == "partial" ? :dash : :dot
            plot!(p, cx, cy, linestyle = style, linecolor = :black)
        end
    end
    # plot physical attack
    colors = Vector{Symbol}(undef, size(df_ap, 1))
    for i in 1:size(df_ap, 1)
        if df_ap.full_coverage[i] > 0
            colors[i] = :green
        elseif df_ap.partial_coverage[i] > 0
            colors[i] = :orange
        else
            colors[i] = :red
        end
    end
    plot!(p, df_ap.x, df_ap.y, seriestype = :scatter, markershape = :x,
        markeralpha = 0.0, markerstrokecolor = colors, markersize = 7,
        markerstrokealpha = 1.0, markerstrokewidth = 1,
        markerstrokestyle = :dash, label = "Physical Attack")
    p
end
