include("../../../lib/factors.jl")
# using ..DistributionCollections
# using ..GaussianDistribution
# using ..Factors
using LaTeXStrings
using Plots
using Distributions
using StatsBase
using Format
using KernelDensity

import Distributions: pdf, rand, Normal
import .GaussianDistribution: Gaussian1D

pdf(d::GaussianDistribution.Gaussian1D, x::Float64) = pdf(Normal(GaussianDistribution.mean(d), sqrt(GaussianDistribution.variance(d))), x)
rand(d::GaussianDistribution.Gaussian1D) = rand(Normal(GaussianDistribution.mean(d), sqrt(GaussianDistribution.variance(d))))
Normal(d::GaussianDistribution.Gaussian1D) = Normal(GaussianDistribution.mean(d), sqrt(GaussianDistribution.variance(d)))
Gaussian1D(d::Normal) = GaussianDistribution.Gaussian1DFromMeanVariance(d.μ, d.σ^2)

function eval(f::Factors.WeightedSumFactor, x::Float64, y::Float64, z::Float64, dirac_std::Float64=1e-1)
    result = pdf(Normal(0.0, dirac_std), z - (x * f.a + y * f.b + f.c))
    # @assert result > 0.0
    return result
end

@enum Term MARGINAL MSGFROM MSGTO

"""
   kde_normal_quotient(kde, normal_dist)
   
Calculate the quotient of a KDE and a normal distribution
"""
function kde_normal_quotient(kde, normal_dist)
    # Evaluate KDE and normal distribution on the KDE's grid
    grid = kde.x
    kde_density = kde.density
    normal_density = pdf.(normal_dist, grid)
    
    # Calculate the quotient
    quotient = kde_density ./ normal_density
    
    # Handle potential division by zero or very small values
    quotient[isinf.(quotient) .| isnan.(quotient)] .= maximum(filter(isfinite, quotient))
    
    # Create a new KDE from the quotient
    quotient_kde = kde(grid, weights=quotient, bandwidth=kde.bandwidth)
    
    return quotient_kde
end

struct WeightedSumFactorParams
    a::Float64
    b::Float64
    c::Float64
    x_mean::Float64
    x_var::Float64
    y_mean::Float64
    y_var::Float64
    z_mean::Float64
    z_var::Float64
end

WeightedSumFactorParams() = WeightedSumFactorParams(1.0, -1.0, 0.0, 1.0, 4.0, 5.0, 4.0, -3.0, 4.0)

"""
    test_sampled_weighted_sum_factor(num_samples::Int64=1_000_000, dirac_std::Float64=1e-1, plot_results=true)

Test the WeightedSumFactor by sampling from the messages and comparing the marginals
"""
function test_sampled_weighted_sum_factor(num_samples::Int64=1_000_000, dirac_std::Float64=1e-1, plot_results=true, term::Term=MARGINAL; params::WeightedSumFactorParams=WeightedSumFactorParams())

    db = DistributionCollections.DistributionBag(GaussianDistribution.Gaussian1D(0, 0))

    x = DistributionCollections.add!(db)
    y = DistributionCollections.add!(db)
    z = DistributionCollections.add!(db)

    db[x] = GaussianDistribution.Gaussian1DFromMeanVariance(params.x_mean, params.x_var)
    db[y] = GaussianDistribution.Gaussian1DFromMeanVariance(params.y_mean, params.y_var)
    db[z] = GaussianDistribution.Gaussian1DFromMeanVariance(params.z_mean, params.z_var)

    f = Factors.WeightedSumFactor(db, x, y, z, params.a, params.b, params.c)


    old_msg_from_x = db[x] / db[f.msg_to_x]
    old_msg_from_y = db[y] / db[f.msg_to_y]
    old_msg_from_z = db[z] / db[f.msg_to_z]

    old_msg_to_x = db[f.msg_to_x]
    old_msg_to_y = db[f.msg_to_y]
    old_msg_to_z = db[f.msg_to_z]

    old_margin_x = db[x]
    old_margin_y = db[y]
    old_margin_z = db[z]

    samples_x = zeros(num_samples)
    samples_y = zeros(num_samples)
    samples_z = zeros(num_samples)
    weights = zeros(num_samples)
    for i in 1:num_samples
        sample_x = rand(old_msg_from_x)
        sample_y = rand(old_msg_from_y)
        sample_z = rand(old_msg_from_z)
        weight = eval(f, sample_x, sample_y, sample_z, dirac_std)
        samples_x[i] = sample_x
        samples_y[i] = sample_y
        samples_z[i] = sample_z
        weights[i] = weight
    end

    Factors.update_msg_to_z!(f)
    Factors.update_msg_to_x!(f)
    Factors.update_msg_to_y!(f)

    sampled_dist_x = Normal(mean(samples_x, Weights(weights)), std(samples_x, Weights(weights)))
    kldiv_x = kldivergence(sampled_dist_x, Normal(db[x]))
    sampled_dist_y = Normal(mean(samples_y, Weights(weights)), std(samples_y, Weights(weights)))
    kldiv_y = kldivergence(sampled_dist_y, Normal(db[y]))
    sampled_dist_z = Normal(mean(samples_z, Weights(weights)), std(samples_z, Weights(weights)))
    kldiv_z = kldivergence(sampled_dist_z, Normal(db[z]))

    new_msg_from_x = db[x] / db[f.msg_to_x]
    new_msg_from_y = db[y] / db[f.msg_to_y]
    new_msg_from_z = db[z] / db[f.msg_to_z]

    sampled_new_msg_to_x = Gaussian1D(sampled_dist_x) / old_msg_from_x
    sampled_new_msg_to_y = Gaussian1D(sampled_dist_y) / old_msg_from_y
    sampled_new_msg_to_z = Gaussian1D(sampled_dist_z) / old_msg_from_z

    sampled_new_msg_from_x = Gaussian1D(sampled_dist_x) / sampled_new_msg_to_x
    sampled_new_msg_from_y = Gaussian1D(sampled_dist_y) / sampled_new_msg_to_y
    sampled_new_msg_from_z = Gaussian1D(sampled_dist_z) / sampled_new_msg_to_z

    if plot_results
        xmin = min(
            GaussianDistribution.mean(db[x]) - 3.0 * sqrt(GaussianDistribution.variance(db[x])),
            GaussianDistribution.mean(db[y]) - 3.0 * sqrt(GaussianDistribution.variance(db[y])),
            GaussianDistribution.mean(db[z]) - 3.0 * sqrt(GaussianDistribution.variance(db[z])),
        )

        xmax = max(
            GaussianDistribution.mean(db[x]) + 3.0 * sqrt(GaussianDistribution.variance(db[x])),
            GaussianDistribution.mean(db[y]) + 3.0 * sqrt(GaussianDistribution.variance(db[y])),
            GaussianDistribution.mean(db[z]) + 3.0 * sqrt(GaussianDistribution.variance(db[z])),
        )

        xs = range(xmin, xmax, length=1000)
        println("xs: $(xs)")
        p = plot(xs, 
            v -> 0.0,
            alpha=0.0,
            xlabel=L"v",
            ylabel=L"p(v)",
            xtickfontsize=12,
            ytickfontsize=12,
            xguidefontsize=14,
            yguidefontsize=14,
            legendfontsize=10,
            label=nothing,
        )

        # plot!(xs,
        #     v -> pdf(sampled_dist_x, v), 
        #     lw=5, 
        #     color=:blue, 
        #     label=L"p(x)",
        #     linestyle=:dot,
        # )

        # plot!(xs,
        #     v -> pdf(sampled_dist_y, v), 
        #     lw=5, 
        #     color=:green, 
        #     label=L"p(y)",
        #     linestyle=:dot,
        # )

        # plot!(xs,
        #     v -> pdf(sampled_dist_z, v), 
        #     lw=5, 
        #     color=:red, 
        #     label=L"p(z)",
        #     linestyle=:dot,
        # )

        if term == MARGINAL
            plot!(xs,
                v -> pdf(db[x], v), 
                lw=5, 
                color=:blue, 
                label=L"p(x)",
                linestyle=:dot,
            )

            plot!(xs,
                v -> pdf(db[y], v), 
                lw=5, 
                color=:green, 
                label=L"p(y)",
                linestyle=:dot,
            )

            plot!(xs,
                v -> pdf(db[z], v), 
                lw=5, 
                color=:red, 
                label=L"p(z)",
                linestyle=:dot,
            )

            plot!(samples_x, 
                weights=weights,
                normalize=:pdf,
                color=:blue, 
                alpha=0.3, 
                label=L"p_{\mathrm{sample}}(x)",
                seriestype=:barhist,
            )

            plot!(samples_y, 
                weights=weights,
                normalize=:pdf,
                color=:green, 
                alpha=0.3, 
                label=L"p_{\mathrm{sample}}(y)",
                seriestype=:barhist,
            )

            plot!(samples_z, 
                weights=weights,
                normalize=:pdf,
                color=:red, 
                alpha=0.3, 
                label=L"p_{\mathrm{sample}}(z)",
                seriestype=:barhist,
            )

        elseif term == MSGFROM
            # plot!(xs,
            #     v -> pdf(old_msg_from_x, v), 
            #     lw=5, 
            #     color=:blue, 
            #     label=L"old\ m_{from X}",
            #     alpha=0.5,
            # )

            # plot!(xs,
            #     v -> pdf(old_msg_from_y, v), 
            #     lw=5, 
            #     color=:green, 
            #     label=L"old\ m_{from Y}",
            #     alpha=0.5,
            # )

            # plot!(xs,
            #     v -> pdf(old_msg_from_z, v), 
            #     lw=5, 
            #     color=:red, 
            #     label=L"old\ m_{from Z}",
            #     alpha=0.5,
            # )

            plot!(xs,
                v -> pdf(new_msg_from_x, v), 
                lw=5, 
                color=:blue, 
                label=L"new\ m_{from X}",
                alpha=0.3, 
            )

            plot!(xs,
                v -> pdf(new_msg_from_y, v), 
                lw=5, 
                color=:green, 
                label=L"new\ m_{from Y}",
                alpha=0.3, 
            )

            plot!(xs,
                v -> pdf(new_msg_from_z, v), 
                lw=5, 
                color=:red, 
                label=L"new\ m_{from Z}",
                alpha=0.3, 
            )

            plot!(xs,
                v -> pdf(Normal(sampled_new_msg_from_x), v),
                color=:blue, 
                # alpha=0.3, 
                linestyle=:dot,
                label=L"p_{\mathrm{sample}}new\ m_{from X}",
                # seriestype=:barhist, 
                lw=3, 
            )

            plot!(xs,
                v -> pdf(Normal(sampled_new_msg_from_y), v),
                color=:green, 
                # alpha=0.3, 
                linestyle=:dot,
                label=L"p_{\mathrm{sample}}new\ m_{from Y}",
                # seriestype=:barhist, 
                lw=3, 
            )

            plot!(xs,
                v -> pdf(Normal(sampled_new_msg_from_z), v),
                color=:red, 
                # alpha=0.3, 
                linestyle=:dot,
                label=L"p_{\mathrm{sample}}new\ m_{from Z}",
                # seriestype=:barhist, 
                lw=3, 
            )

        elseif term == MSGTO
            plot!(xs,
                v -> pdf(db[f.msg_to_x], v), 
                lw=5, 
                color=:blue, 
                label=L"new\ m_{to X}",
                alpha=0.3,
            )

            plot!(xs,
                v -> pdf(db[f.msg_to_y], v), 
                lw=5, 
                color=:green, 
                label=L"new\ m_{to Y}",
                alpha=0.3,
            )

            plot!(xs,
                v -> pdf(db[f.msg_to_z], v), 
                lw=5, 
                color=:red, 
                label=L"new\ m_{to Z}",
                alpha=0.3,
            )

            plot!(xs,
                v -> pdf(Normal(sampled_new_msg_to_x), v),
                color=:blue, 
                # alpha=0.3, 
                label=L"p_{\mathrm{sample}}new\ m_{to X}",
                # seriestype=:barhist,
                linestyle=:dot, 
                lw=3, 
            )

            plot!(xs,
                v -> pdf(Normal(sampled_new_msg_to_y), v),
                color=:green, 
                # alpha=0.3, 
                label=L"p_{\mathrm{sample}}new\ m_{to Y}",
                # seriestype=:barhist,
                linestyle=:dot, 
                lw=3, 
            )

            plot!(xs,
                v -> pdf(Normal(sampled_new_msg_to_z), v),
                color=:red, 
                # alpha=0.3, 
                label=L"p_{\mathrm{sample}}new\ m_{to Z}",
                # seriestype=:barhist,
                linestyle=:dot, 
                lw=3, 
            )
        end

        display(p)
    end

    println("KLDiv(X), KLDiv(Y), KLDiv(Z): ", kldiv_x, ", ", kldiv_y, ", ", kldiv_z)
end

"""
    gridsearch_sampled_weighted_sum_factor()

Grid search on number of samples and standard deviation used to approximate dirac delta
"""
function gridsearch_sampled_weighted_sum_factor()
    for num_samples in [1_000_000, 10_000_000, 100_000_000]
        for dirac_std in [1e-3, 1e-2, 1e-1, 1.0]
            println("Samples: ", format(num_samples, commas=true), ", Std: ", dirac_std)
            test_sampled_weighted_sum_factor(num_samples, dirac_std, false)
        end
    end
end

p = WeightedSumFactorParams(1.0, -1.0, 0.0, 1.0, 4.0, 4.0, 4.0, 0.0, 4.0)
test_sampled_weighted_sum_factor(1_000_000, 1e-1, true, MARGINAL; params=p)