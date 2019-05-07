function reshape_product(A::AbstractArray)
    k = 0
    v = Vector{Vector{Int64}}(undef, prod(size(A)))
    for i in eachindex(A)
        v[i] = [A[i]...]
    end
    return sort!(v)
end

function matchrow(a::AbstractVector, B::AbstractArray)
    findfirst(i -> all(j -> a[j] == B[i,j], 1:size(B,2)), 1:size(B,1))
end

function circlecoord(x::Float64, y::Float64, r::Float64)
    radians = -0.1:0.1:2pi
    cx = [cos(rad)r + x for rad in radians]
    cy = [sin(rad)r + y for rad in radians]
    cx, cy
end

function rerr_approx(u::Vector)
    lb = round(Int, 0.9 * length(u))
    u_approx = mean(u[lb:end])
    um = cumsum(u) ./ collect(1:T)
    u_rerr_approx = abs.(um .- u_approx) ./ abs.(u_approx)
    return u_rerr_approx
end

rerr(u::Number, u_true::Number) = abs(u - u_true) / abs(u_true)

tosecond(t::T) where {T} = t / convert(T, Dates.Second(1))
tominute(t::T) where {T} = t / convert(T, Dates.Minute(1))
tohour(t::T) where {T} = t / convert(T, Dates.Hour(1))
