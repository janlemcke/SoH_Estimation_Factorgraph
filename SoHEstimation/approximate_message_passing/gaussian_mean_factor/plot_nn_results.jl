using Flux
using JLD2
using ProgressMeter
using Random
using Plots


function update_msg_to_x(fromymean, fromyvar, β2)
    msg_from_y = GaussianDistribution.Gaussian1DFromMeanVariance(fromymean, fromyvar)
    c = 1.0 + β2 * msg_from_y.rho

    newMsg = GaussianDistribution.Gaussian1D(msg_from_y.tau / c, msg_from_y.rho / c)
    
    return newMsg
end

function update_marginal_x(fromxmean, fromxvar, fromymean, fromyvar, β2)
    return GaussianDistribution.Gaussian1DFromMeanVariance(fromxmean, fromxvar) * update_msg_to_x(fromymean, fromyvar, β2)
end

function update_msg_to_y(fromxmean, fromxvar, β2)
    msg_from_x = GaussianDistribution.Gaussian1DFromMeanVariance(fromxmean, fromxvar)
    c = 1.0 + β2 * msg_from_x.rho

    newMsg = GaussianDistribution.Gaussian1D(msg_from_x.tau / c, msg_from_x.rho / c)
    return newMsg
end

function update_marginal_y(fromxmean, fromxvar, fromymean, fromyvar, β2)
    return GaussianDistribution.Gaussian1DFromMeanVariance(fromymean, fromyvar) * update_msg_to_y(fromxmean, fromxvar, β2)
end

function update_marginal(variable::Variable, fromxmean, fromxvar, fromymean, fromyvar, β2)
    if variable == X
        return update_marginal_x(fromxmean, fromxvar, fromymean, fromyvar, β2)
    elseif variable == Y
        return update_marginal_y(fromxmean, fromxvar, fromymean, fromyvar, β2)
    else
        error("Variable not supported")
    end
end

function update_marginals(fromxmean, fromxvar, fromymean, fromyvar, β2; natural_parameters=true)
    x = update_marginal_x(fromxmean, fromxvar, fromymean, fromyvar, β2)
    y = update_marginal_y(fromxmean, fromxvar, fromymean, fromyvar, β2)

    if natural_parameters
        return [x.tau, x.rho, y.tau, y.rho]
    else
        return [GaussianDistribution.mean(x), GaussianDistribution.variance(x), GaussianDistribution.mean(y), GaussianDistribution.variance(y)]
    end
end

function update_msg_to(variable::Variable, fromxmean, fromxvar, fromymean, fromyvar, β2)
    if variable == X
        return update_msg_to_x(fromymean, fromyvar, β2)
    elseif variable == Y
        return update_msg_to_y(fromxmean, fromxvar, β2)
    else
        error("Variable not supported")
    end
end

function update_messages_to_variables(fromxmean, fromxvar, fromymean, fromyvar, β2; natural_parameters=true)
    x = update_msg_to_x(fromymean, fromyvar, β2)
    y = update_msg_to_y(fromxmean, fromxvar, β2)

    if natural_parameters
        return [x.tau, x.rho, y.tau, y.rho]
    else
        return [GaussianDistribution.mean(x), GaussianDistribution.variance(x), GaussianDistribution.mean(y), GaussianDistribution.variance(y)]
    end
end

function normalize_sample(sample, norms)
    return [(sample[dim] - norms[dim][:mean]) / sqrt(norms[dim][:var]) for dim in eachindex(sample)]
end

function plot_results_mean(; modelpath="SoHEstimation/approximate_message_passing/weighted_sum_factor/models/wsf_weights_nn_update_X_N16_000_norm.jld2", variable=X, moment=1, change_dims=[3, 5], sample_default=[5.0, 2.0, 10.0, 4.0, 2.5], plotrange=-10:0.1:10, output_index=0, loss_choice="mse")
    @assert moment in [1, 2]
    f32sample_default = map(Float32, sample_default)
    stored = JLD2.load(modelpath)
    model = stored["model"]
    norms = Vector{Dict{Symbol,Float32}}(stored["norms"])

    function get_sample(change_var_1, change_var_2)
        sample = deepcopy(f32sample_default)
        sample[change_dims[1]] = change_var_1
        sample[change_dims[2]] = change_var_2
        return sample
    end

    function zs_nn(change_var_1, change_var_2)
        sample = get_sample(change_var_1, change_var_2)
        sample = remove_variable(variable, sample)
        sample = isnothing(norms) ? sample : normalize_sample(sample, norms)
        updated_message = model(sample)
        if output_index == 0
            return updated_message[moment]
        else
            return updated_message[1]
        end
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

    plotlyjs()

    p1 = surface(xs, ys, zs_nn_values, title="WSF NN vs. Analytical. Variable: $(variable), Moment: $(moment), Output index: $(output_index), Loss choice: $(loss_choice)",
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
    # display(final_plot)
    # readline()
    return final_plot
end