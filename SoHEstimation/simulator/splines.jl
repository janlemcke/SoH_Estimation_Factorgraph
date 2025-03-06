struct SplineSet
    a::Float64
    b::Float64
    c::Float64
    d::Float64
    x::Float64
end

function spline(x::Vector{Float64}, y::Vector{Float64})::Vector{SplineSet}
    n = length(x) - 1
    a = y
    b = Vector{Float64}(undef, n)
    d = Vector{Float64}(undef, n)
    h = Vector{Float64}()

    for i in 1:n
        push!(h, x[i + 1] - x[i])
    end

    alpha = [0.0]
    for i in 2:n
        push!(alpha, 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1])
    end

    c = Vector{Float64}(undef, n + 1)
    l = Vector{Float64}(undef, n + 1)
    mu = Vector{Float64}(undef, n + 1)
    z = Vector{Float64}(undef, n + 1)
    l[1] = 1.0
    mu[1] = 0.0
    z[1] = 0.0

    for i in 2:n
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    end

    l[n + 1] = 1.0
    z[n + 1] = 0.0
    c[n + 1] = 0.0

    for j in n:-1:1
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / 3 / h[j]
    end

    return [SplineSet(a[i], b[i], c[i], d[i], x[i]) for i in 1:n]
end