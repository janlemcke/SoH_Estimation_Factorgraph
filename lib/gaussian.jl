# Library for 1D Gaussian messages and distribution
#
# 2023 by Ralf Herbrich
# Hasso-Plattner Institute

module GaussianDistribution
export Gaussian1D, Gaussian1DFromMeanVariance, mean, variance, absdiff, logNormProduct, logNormRatio
"""
Data structure that captures the state of an normalized 1D Gaussian. 
In this represenation, we are storing the precision times mean (tau) and the 
precision (rho). This representations allows for numerically stable products of 
1D-Gaussians.

Should contain two variables:
tau and rho, both Floats, to store the natural parameters of the gaussian.
"""
struct Gaussian1D
    tau::Float64            # the precision mean, tau = μ/σ^2 = μ * rho, is the precision adjusted mean
    rho::Float64            # the precision, rho = 1/σ^2, is the inverse of the variance

    # default constructor
    Gaussian1D(tau, rho) =
        (rho < 0) ? error("precision of a Gaussian must be non-negative. (rho = $rho)") :
        new(promote(tau, rho)...)
end
# Initializes a standard Gaussian 
Gaussian1D() = Gaussian1D(0, 1)

"""
    Gaussian1DFromMeanVariance(μ,σ2)

Initializes a Gaussian from mean and variance.
"""
Gaussian1DFromMeanVariance(μ, σ2) = Gaussian1D(μ / σ2, 1 / σ2)

"""
    mean(g)

Returns the mean of the 1D-Gaussian
```julia-repl
julia> mean(Gaussian1D(1,2))
0.5

julia> mean(Gaussian1DFromMeanVariance(1,2))
1.0
```
"""
mean(g::Gaussian1D) = (g.rho == 0.0) && !(isnan(g.rho) || isnan(g.tau)) ? 0.0 :  g.tau / g.rho

"""
    variance(g)

Returns the variance of the 1D-Gaussian 
```julia-repl
julia> variance(Gaussian1D(1,2))
0.5

julia> variance(Gaussian1DFromMeanVariance(1,2))
2.0
```
"""
variance(g::Gaussian1D) = 1.0 / g.rho


"""
    absdiff(g1,g2)

Computes the absolute difference of `g1` and `g2` in terms of tau and rho
# Examples
```julia-repl
julia> absdiff(Gaussian1D(0,1),Gaussian1D(0,2))
1.0

julia> absdiff(Gaussian1D(0,1),Gaussian1D(0,3))
1.4142135623730951
```
"""
absdiff(g1::Gaussian1D, g2::Gaussian1D) =
    max(abs(g1.tau - g2.tau), sqrt(abs(g1.rho - g2.rho)))

"""
    *(g1,g2)

Multiplies two 1D Gaussians together and re-normalizes them
# Examples
```julia-repl
julia> Gaussian1D() * Gaussian1D()
μ = 0.0, σ = 0.7071067811865476
```
"""
function Base.:*(g1::Gaussian1D, g2::Gaussian1D)
    return (Gaussian1D(g1.tau + g2.tau, g1.rho + g2.rho))
end

"""
    /(g1,g2)

Divides two 1D Gaussians from each other
# Examples
```julia-repl
julia> Gaussian1D(0,1) / Gaussian1D(0,0.5)
μ = 0.0, σ = 1.4142135623730951
```
"""
function Base.:/(g1::Gaussian1D, g2::Gaussian1D)
    return (Gaussian1D(g1.tau - g2.tau, g1.rho - g2.rho))
end

"""
    logNormProduct(g1,g2)

Computes the log-normalization constant of a multiplication of `g1` and `g2` (the end of the equation ;))

It should be 0 if both rho variables are 0.
# Examples
```julia-repl
julia> logNormProduct(Gaussian1D() * Gaussian1D())
c = 0.28209479177387814
```
"""
function logNormProduct(g1::Gaussian1D, g2::Gaussian1D)
    if (g1.rho == 0.0 || g2.rho == 0.0)
        return (0.0)
    else
        σ2Sum = variance(g1) + variance(g2)
        μDiff = mean(g1) - mean(g2)
        return (-0.5 * (log(2 * π * σ2Sum) + μDiff * μDiff / σ2Sum))
    end
end

"""
    logNormRatio(g1,g2)

Computes the log-normalization constant of a division of `g1` with `g2` (the end of the equation ;))

It should be 0 if both rho variables are 0.
# Examples
```julia-repl
julia> logNormRatio(Gaussian1D(0,1) / Gaussian1D(0,0.5))
5.013256549262001
```
"""
function logNormRatio(g1::Gaussian1D, g2::Gaussian1D)
    if (g1.rho == 0.0 || g2.rho == 0.0)
        return (0.0)
    else
        g2Var = variance(g2)
        σ2Diff = g2Var - variance(g1)
        if (σ2Diff == 0.0)
            return (0.0)
        else
            μDiff = mean(g1) - mean(g2)
            return (log(g2Var) + 0.5 * (log(2 * π / σ2Diff) + μDiff * μDiff / σ2Diff))
        end
    end
end

"""
    show(io,g)

Pretty-prints a 1D Gaussian
"""
function Base.show(io::IO, g::Gaussian1D)
    if (g.rho == 0.0) && !(isnan(g.rho) || isnan(g.tau))
        print(io, "μ = 0, σ = Inf")
    else
        print(io, "μ = ", mean(g), ", σ = ", sqrt(variance(g)))
    end
end

function squared_diff(approx::Gaussian1D, exact::Gaussian1D)
    mean_diff = GaussianDistribution.mean(approx) - GaussianDistribution.mean(exact)
    var_exact = GaussianDistribution.variance(exact)
    var_approx = GaussianDistribution.variance(approx)

    if GaussianDistribution.mean(approx) == Inf || !isfinite(GaussianDistribution.mean(approx))
        error("Approximate mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if GaussianDistribution.mean(exact) == Inf || !isfinite(GaussianDistribution.mean(exact))
        error("Exact mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_exact == Inf || !isfinite(var_exact)
        error("Exact variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_approx == Inf|| !isfinite(var_exact)
        error("Sampled variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end
    variance_diff = var_approx - var_exact

    rho_diff = approx.rho - exact.rho
    tau_diff = approx.tau - exact.tau

    return mean_diff^2, variance_diff^2, rho_diff^2, tau_diff^2
end

function absolute_error(approx::Gaussian1D, exact::Gaussian1D)
    mean_diff = abs(GaussianDistribution.mean(approx) - GaussianDistribution.mean(exact))
    var_exact = GaussianDistribution.variance(exact)
    var_approx = GaussianDistribution.variance(approx)

    if GaussianDistribution.mean(approx) == Inf || !isfinite(GaussianDistribution.mean(approx))
        error("Approximate mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if GaussianDistribution.mean(exact) == Inf || !isfinite(GaussianDistribution.mean(exact))
        error("Exact mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_exact == Inf || !isfinite(var_exact)
        error("Exact variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_approx == Inf|| !isfinite(var_exact)
        error("Sampled variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    variance_diff = abs(var_approx - var_exact)

    rho_diff = abs(approx.rho - exact.rho)
    tau_diff = abs(approx.tau - exact.tau)

    return mean_diff, variance_diff, rho_diff, tau_diff
end


function absolute_percentage_error(approx::Gaussian1D, exact::Gaussian1D)

    if GaussianDistribution.mean(exact) == 0.0
        error("Exact mean is 0.0 given approx: ", approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)   
    end

    mean_term = abs((GaussianDistribution.mean(exact) - GaussianDistribution.mean(approx)) / GaussianDistribution.mean(exact))
    var_exact = GaussianDistribution.variance(exact)
    var_approx = GaussianDistribution.variance(approx)
 
    if GaussianDistribution.mean(approx) == Inf || !isfinite(GaussianDistribution.mean(approx))
        error("Approximate mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if GaussianDistribution.mean(exact) == Inf || !isfinite(GaussianDistribution.mean(exact))
        error("Exact mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_exact == Inf || !isfinite(var_exact)
        error("Exact variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_approx == Inf|| !isfinite(var_exact)
        error("Sampled variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    variance_term = abs((GaussianDistribution.variance(exact) - GaussianDistribution.variance(approx)) / GaussianDistribution.variance(exact))

    rho_term = abs((exact.rho - approx.rho) / exact.rho)
    tau_term = abs((exact.tau - approx.tau) / exact.tau)

    return mean_term, variance_term, rho_term, tau_term
end

"""
    absolute_percentage_error_soft_fail(approx::Gaussian1D, exact::Gaussian1D)
Soft Fail: Returns NaN for exact mean == 0, instead of throwing an error.
"""
function absolute_percentage_error_soft_fail(approx::Gaussian1D, exact::Gaussian1D)

    var_exact = GaussianDistribution.variance(exact)
    var_approx = GaussianDistribution.variance(approx)
 
    if GaussianDistribution.mean(approx) == Inf || !isfinite(GaussianDistribution.mean(approx))
        error("Approximate mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if GaussianDistribution.mean(exact) == Inf || !isfinite(GaussianDistribution.mean(exact))
        error("Exact mean is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_exact == Inf || !isfinite(var_exact)
        error("Exact variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    if var_approx == Inf|| !isfinite(var_exact)
        error("Sampled variance is Inf:" , approx, " rho: ", approx.rho, " tau: ", approx.tau, " exact message: ", exact, " rho: ", exact.rho, " tau: ", exact.tau)
    end

    variance_term = abs((GaussianDistribution.variance(exact) - GaussianDistribution.variance(approx)) / GaussianDistribution.variance(exact))

    rho_term = abs((exact.rho - approx.rho) / exact.rho)

    if GaussianDistribution.mean(exact) == 0.0
        mean_term = NaN
        tau_term = NaN
    else
        mean_term = abs((GaussianDistribution.mean(exact) - GaussianDistribution.mean(approx)) / GaussianDistribution.mean(exact))
        tau_term = abs((exact.tau - approx.tau) / exact.tau)
    end

    return mean_term, variance_term, rho_term, tau_term
end

end