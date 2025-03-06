using Flux
using JLD2
using ProgressMeter
using Random
using Plots
using LaTeXStrings

function update_msg_to_x(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    if (!isfinite(fromymean) && !isfinite(fromyvar)) || (!isfinite(fromzmean) && !isfinite(fromzvar))
        # if one of messages from the other variables is still a uniform distribution (i.e. has not changed since being intialized), do not update the marginal
        return GaussianDistribution.Gaussian1D(0,0)
    else
        return GaussianDistribution.Gaussian1DFromMeanVariance(
            fromzmean / a - b / a * fromymean - c / a,
            fromzvar / (a * a) + b * b / (a * a) * fromyvar,
        )
    end
end

function update_marginal_x(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    # recognize cases where the message from the current variable is still a uniform distribution (i.e. has not changed since being intialized). 
    # if original input was in natural parameters, transforming to mean and variance will result in NaN/Inf.
    fromx = !isfinite(fromxmean) && !isfinite(fromxvar) ? GaussianDistribution.Gaussian1D(0,0) : GaussianDistribution.Gaussian1DFromMeanVariance(fromxmean, fromxvar)

    return fromx * update_msg_to_x(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
end

function update_msg_to_y(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    if (!isfinite(fromxmean) && !isfinite(fromxvar)) || (!isfinite(fromzmean) && !isfinite(fromzvar))
        # if one of messages from the other variables is still a uniform distribution (i.e. has not changed since being intialized), do not update the marginal
        return GaussianDistribution.Gaussian1D(0,0)
    else
        return GaussianDistribution.Gaussian1DFromMeanVariance(
            fromzmean / b - a / b * fromxmean - c / b,
            fromzvar / (b * b) + a * a / (b * b) * fromxvar,
        )
    end
end

function update_marginal_y(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    # recognize cases where the message from the current variable is still a uniform distribution (i.e. has not changed since being intialized). 
    # if original input was in natural parameters, transforming to mean and variance will result in NaN/Inf.
    fromy = !isfinite(fromymean) && !isfinite(fromyvar) ? GaussianDistribution.Gaussian1D(0,0) : GaussianDistribution.Gaussian1DFromMeanVariance(fromymean, fromyvar)

    return fromy * update_msg_to_y(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
end

function update_msg_to_z(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    if (!isfinite(fromxmean) && !isfinite(fromxvar)) || (!isfinite(fromymean) && !isfinite(fromyvar))
        # if one of messages from the other variables is still a uniform distribution (i.e. has not changed since being intialized), do not update the marginal
        return GaussianDistribution.Gaussian1D(0,0)
    else
        return GaussianDistribution.Gaussian1DFromMeanVariance(
            a * fromxmean + b * fromymean + c,
            a * a * fromxvar + b * b * fromyvar,
        )
    end
end

function update_marginal_z(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    # recognize cases where the message from the current variable is still a uniform distribution (i.e. has not changed since being intialized). 
    # if original input was in natural parameters, transforming to mean and variance will result in NaN/Inf.
    fromz = !isfinite(fromzmean) && !isfinite(fromzvar) ? GaussianDistribution.Gaussian1D(0,0) : GaussianDistribution.Gaussian1DFromMeanVariance(fromzmean, fromzvar)

    return fromz * update_msg_to_z(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
end

function update_marginal(variable::WeightedSumFactorGeneration.Variable, fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    if variable == X
        return update_marginal_x(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    elseif variable == Y
        return update_marginal_y(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    elseif variable == Z
        return update_marginal_z(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    else
        error("Variable not supported")
    end
end

function update_marginals(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c; natural_parameters=true)
    x = update_marginal_x(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    y = update_marginal_y(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    z = update_marginal_z(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)

    if natural_parameters
        return [x.tau, x.rho, y.tau, y.rho, z.tau, z.rho]
    else
        return [GaussianDistribution.mean(x), GaussianDistribution.variance(x), GaussianDistribution.mean(y), GaussianDistribution.variance(y), GaussianDistribution.mean(z), GaussianDistribution.variance(z)]
    end
end

function update_msg_to(variable::WeightedSumFactorGeneration.Variable, fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    if variable == X
        return update_msg_to_x(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    elseif variable == Y
        return update_msg_to_y(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    elseif variable == Z
        return update_msg_to_z(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    else
        error("Variable not supported")
    end
end

function update_messages_to_variables(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c; natural_parameters=true)
    x = update_msg_to_x(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    y = update_msg_to_y(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)
    z = update_msg_to_z(fromxmean, fromxvar, fromymean, fromyvar, fromzmean, fromzvar, a, b, c)

    if natural_parameters
        return [x.tau, x.rho, y.tau, y.rho, z.tau, z.rho]
    else
        return [GaussianDistribution.mean(x), GaussianDistribution.variance(x), GaussianDistribution.mean(y), GaussianDistribution.variance(y), GaussianDistribution.mean(z), GaussianDistribution.variance(z)]
    end
end

function normalize_sample(sample, norms)
    return [(sample[dim] - norms[dim][:mean]) / sqrt(norms[dim][:var]) for dim in eachindex(sample)]
end

function plot_results_mean(; modelpath="SoHEstimation/approximate_message_passing/weighted_sum_factor/models/wsf_weights_nn_update_X_N16_000_norm.jld2", variable=X, moment=1, change_dims=[3, 5], sample_default=[1.0, 4.0, 10.0, 4.0, 0.0, 4.0, 1.0, -1.0, 0.0], plotrange=-10:0.1:10, scale_third_variable=true)
    @assert moment in [1, 2]
    f32sample_default = map(Float32, sample_default)
    stored = JLD2.load(modelpath)
    model = stored["best_model"]
    norms = Vector{Dict{Symbol,Float32}}(stored["norms"])

    function get_sample(change_var_1, change_var_2)
        sample = deepcopy(f32sample_default)
        sample[change_dims[1]] = change_var_1
        sample[change_dims[2]] = change_var_2

        # # not needed anymore because we are now plotting messages instead of marginals
        # # since sampling does not work if the mean of a prior distribution of a message from a variable is more than 15 off from the value implied by the other messages, 
        # # we scale the mean to match the differnce between implied and actual value that is present in the default sample. if this is not done, the nn prediction be far off.
        # # to see this, set scale_third_variable=false and plot a value range > 15 for the means of (the message from) two variables, i.e., dimensions 3, 5 or 1 of the input.
        # if scale_third_variable
        #     if !(5 in change_dims) && (3 in change_dims || 1 in change_dims)
        #         diff = f32sample_default[5] - (f32sample_default[1] * f32sample_default[7] + f32sample_default[3] * f32sample_default[8] + f32sample_default[9])
        #         sample[5] = (sample[1] * sample[7] + sample[3] * sample[8] + sample[9]) + diff
        #     elseif !(1 in change_dims) && (3 in change_dims || 5 in change_dims)
        #         diff = f32sample_default[1] - ((f32sample_default[5] - f32sample_default[3] * f32sample_default[8] - f32sample_default[9]) / f32sample_default[7])
        #         sample[1] = ((sample[5] - sample[3] * sample[8] - sample[9]) / sample[7]) + diff
        #     elseif !(3 in change_dims) && (5 in change_dims || 1 in change_dims)
        #         diff = f32sample_default[3] - ((f32sample_default[5] - f32sample_default[1] * f32sample_default[7] - f32sample_default[9]) / f32sample_default[8])
        #         sample[3] = ((sample[5] - sample[1] * sample[7] - sample[9]) / sample[8]) + diff
        #     end
        # end
        return sample
    end

    function zs_nn(change_var_1, change_var_2)
        sample = get_sample(change_var_1, change_var_2)
        sample = isnothing(norms) ? sample : normalize_sample(sample, norms)
        sample = remove_variable(variable, sample)
        updated_message = model(sample)
        return updated_message[moment]
    end

    function zs_factor(change_var_1, change_var_2)
        sample = get_sample(change_var_1, change_var_2)
        updated_message = update_msg_to(variable, sample...)
        return moment == 1 ? GaussianDistribution.mean(updated_message) : GaussianDistribution.variance(updated_message)
    end

    f32_plotrange = map(Float32, plotrange)
    xs = f32_plotrange
    ys = f32_plotrange
    zs_factor_values = [zs_factor(x, y) for x in xs, y in ys]
    zs_nn_values = [zs_nn(x, y) for x in xs, y in ys]

    # color_values = zs_nn_values - zs_factor_values
    # colorscale = [
    #     [-1.0, "rgb(0,0,255)"],        # Blue for negative values
    #     [1.0, "rgb(0,0,255)"],        # Blue for negative values
    #     [1.0, "rgb(255,0,0)"],  # Red for positive zero
    #     [2.0, "rgb(255,0,0)"]         # Red for positive values
    # ]

    plotlyjs()

    p1 = surface(xs, ys, zs_nn_values, title="WSF NN vs. Analytical.",
        # surfacecolor=color_values, colorscale=colorscale, colorbar_title="nn - analytical",
        xlabel="Change in dim $(change_dims[1])", ylabel="Change in dim $(change_dims[2])", zlabel="Change in dim $(dimension(variable, moment))",
        label="NN", legend=:topleft, alpha=0.8,
        color=:blue
    )

    surface!(
        xs, ys, zs_factor_values,
        label="Analytical", color=:green, colorbar=false,# alpha=0.8, 
    )

    # surface!(
    #     xs, ys, color_values, #st = [:surface], 
    #     label="Difference", alpha=0.5, colorbar=false,# color=:green,
    #     surfacecolor=zs_nn_values, colorscale=colorscale,
    # )

    footnote = "Dims: 1 - mean msgfrom x, 2 - var msgfrom x, 3 - mean msgfrom y, \n4 - var msgfrom y, 5 - mean msgfrom z, 6 - var msgfrom z, 7 - a , 8 - b, 9 - c"
    p2 = plot(
        annotation=(0.5, 0.5, text(footnote, :center, 10, :grey)),
        showaxis=false, ticks=nothing, legend=false
    )
    layout = @layout [a; b{0.1h}]
    final_plot = plot(p1, p2, layout=layout, size=(800, 700))
    display(final_plot)
    readline()
end

function plot_analytical(; variable=X, moment=1, change_dims=[3, 5], sample_default=[1.0, 4.0, 10.0, 4.0, 0.0, 4.0, 1.0, -1.0, 0.0], plotrange=-1000:100:1000, nat_params=false)

    @assert moment in [1, 2]
    # f32sample_default = map(Float32, sample_default)

    function get_sample(change_var_1, change_var_2)
        sample = deepcopy(sample_default)
        sample[change_dims[1]] = change_var_1
        sample[change_dims[2]] = change_var_2

        return sample
    end

    isvar1 = change_dims[1] in [2, 4, 6]
    isvar2 = change_dims[2] in [2, 4, 6]

    nanclamp(x, lb, ub) = ub < x < lb ? NaN : x

    function zs_factor(change_var_1, change_var_2)
        if isvar1 && change_var_1 < 0
            return NaN
        end
        if isvar2 && change_var_2 < 0
            return NaN
        end
        # if isvar1 && change_var_1 == 0 change_var_1 = 1e-6 end
        # if isvar2 && change_var_2 == 0 change_var_2 = 1e-6 end
        # change_var_1 = isvar1 ? clamp(change_var_1, 1e-10, Inf) : change_var_1
        # change_var_2 = isvar2 ? clamp(change_var_2, 1e-10, Inf) : change_var_2
        sample = get_sample(change_var_1, change_var_2)
        updated_message = update_msg_to(variable, sample...)
        return nat_params ? (moment == 1 ? updated_message.tau : updated_message.rho) : (moment == 1 ? GaussianDistribution.mean(updated_message) : GaussianDistribution.variance(updated_message))
    end

    # f32_plotrange = map(Float32, plotrange)

    xs = plotrange # change_dims[1] in [2,4,6] ? (0.00001:step(plotrange):last(plotrange)) : plotrange
    ys = plotrange # change_dims[2] in [2,4,6] ? (0.00001:step(plotrange):last(plotrange)) : plotrange
    zs_factor_values = [
        try
            zs_factor(x, y)
        catch e
            print(e, " ", x, " ", y)
        end for x in xs, y in ys
    ]

    println("zs count:", length(zs_factor_values), ", mean: ", StatsBase.mean(skipmissing(zs_factor_values)), ", zs std: ", StatsBase.std(skipmissing(zs_factor_values)), ", zs min: ", minimum(skipmissing(zs_factor_values)), ", zs max: ", maximum(skipmissing(zs_factor_values)), ", zs nan count: ", sum(isnan.(zs_factor_values)), ", zs inf count: ", sum(isinf.(zs_factor_values)), ", zs median: ", StatsBase.median(skipmissing(zs_factor_values)))
    param(i) = nat_params ? i == 1 ? "τ" : "ρ" : i == 1 ? "μ" : "σ^2"

    function string_from_dim(dim)
        if dim == 1
            return "$(param(1))_x"
        elseif dim == 2
            return "$(param(2))_x"
        elseif dim == 3
            return "$(param(1))_y"
        elseif dim == 4
            return "$(param(2))_y"
        elseif dim == 5
            return "$(param(1))_z"
        elseif dim == 6
            return "$(param(2))_z"
        elseif dim == 7
            return "a"
        elseif dim == 8
            return "b"
        elseif dim == 9
            return "c"
        end
    end

    function input_string_from_dim(dim)
        if dim == 1
            return "μ_x"
        elseif dim == 2
            return "σ^2_x"
        elseif dim == 3
            return "μ_y"
        elseif dim == 4
            return "σ^2_y"
        elseif dim == 5
            return "μ_z"
        elseif dim == 6
            return "σ^2_z"
        elseif dim == 7
            return "a"
        elseif dim == 8
            return "b"
        elseif dim == 9
            return "c"
        end
    end

    plotlyjs()

    p1 = surface(
        xs, ys, zs_factor_values,
        label="Analytical", color=:bamako, colorbar=false,# alpha=0.8, 
        xlabel="$(input_string_from_dim(change_dims[1]))",
        ylabel="$(input_string_from_dim(change_dims[2]))",
        zlabel="$(moment == 1 ? "$(param(1))_" : "$(param(2))_")$(string(variable))",
    )

    # footnote = latexstring(
    #     """
    #         $(param(1))_x = $(mean(X, sample_default)), $(param(2))_x = $(variance(X, sample_default)), 
    #         $(param(1))_y = $(mean(Y, sample_default)), $(param(2))_y = $(variance(Y, sample_default)), $(param(1))_z = $(mean(Z, sample_default)),
    #          = $(variance(Z, sample_default)), a = $(a(sample_default)), b = $(b(sample_default)), c = $(c(sample_default))
    #         _z = $(variance(Z, sample_default)), a = $(a(sample_default)), b = $(b(sample_default)), c = $(c(sample_default))
    #     """
    # )
    footnote = """
                   default:    μ_x = $(round(mean(X, sample_default), digits=3)),    σ^2_x = $(round(variance(X, sample_default), digits=3)), 
                   μ_y = $(round(mean(Y, sample_default), digits=3)),    σ^2_y = $(round(variance(Y, sample_default), digits=3)),
               """
    footnote2 = """
                    μ_z = $(round(mean(Z, sample_default), digits=3)),
                    σ^2_z = $(round(variance(Z, sample_default), digits=3)),    a = $(round(a(sample_default), digits=3)), b = $(round(b(sample_default), digits=3)), c = $(round(c(sample_default), digits=3))
                """

    p2 = plot(
        title="x: $(input_string_from_dim(change_dims[1]))     y: $(input_string_from_dim(change_dims[2]))     z: $(moment == 1 ? "$(param(1))_" : "$(param(2))_")$(string(variable))",
        annotation=(0.5, 0.5, text(footnote * footnote2, :center, 5, :grey, fontzise=1)),
        showaxis=false, ticks=nothing, legend=false,
        # hovertemplate= ["x: %{x}","y: %{y}","value: %{z}"]
    )
    layout = @layout [a; b{0.1h}]
    final_plot = plot(p1, p2, layout=layout, size=(800, 700))
    # display(final_plot)
    # readline()

    # p1 = surface(
    #     xs, ys, zs_factor_values,
    #     label="Analytical", color=:green, colorbar=false,# alpha=0.8, 
    # )
    # display(p1)
end

function dense_near_zero(start::Number, stop::Number, num_points::Int)
    # Create a linearly spaced array from 0 to 1
    t = range(0, 1, length=num_points)

    # Apply a function to increase density near zero
    x = (exp.(3t) .- 1) / (exp(3) - 1)

    # Scale to the desired range
    return start .+ (stop - start) .* x
end