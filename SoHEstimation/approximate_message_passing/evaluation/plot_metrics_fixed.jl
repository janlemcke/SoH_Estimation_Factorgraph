using CSV
using DataFrames
using Plots
using StatsBase
using Printf
using LaTeXStrings
using Measures

# Define the possible values
sources = ["Prediction_vs_Exact", "Sampled_vs_Exact"]
factors = ["gmf", "wsf"]

# Function to get valid combinations
function get_valid_combinations(factor)
    if factor == "gmf"
        fixed = ["targets_X", "targets_Y"]
        variables = ["X", "Y", "beta^2"]
    else  # wsf
        fixed = ["targets_X", "targets_Y", "targets_Z"]
        variables = ["X", "Y", "Z", "a_b"]
    end
    
    # Filter out invalid combinations (where Fixed target matches Variable)
    combinations = []
    for f in fixed
        for v in variables
            if !(lowercase(replace(f, "targets_" => "")) == lowercase(v))
                push!(combinations, (f, v))
            end
        end
    end
    return combinations
end

function smart_round(x)
    if x == 0
        return "0.00"
    elseif isnan(x)
        return "NaN"
    end
    
    # Find first non-zero decimal place
    place = floor(Int, -log10(abs(x)))
    if place <= -2  # If number is >= 100
        value = @sprintf("%.1f", x)
    elseif place >= 2  # If number is very small
        value = string(round(x, digits=place+2))
    else
        value = @sprintf("%.2f", x)
    end
    return value
end

function plot_2D_variable(df, source, fixed, variable, factor, metric)
    # Get unique gridpoints
    filtered_df = filter(row -> row.Source == source && row.Fixed == fixed && row.Variable == variable && row.Factor == factor, df)
    gridpoints = unique(filtered_df.Gridpoint)
    x_vals = sort(unique([eval(Meta.parse(gp))[1] for gp in gridpoints]))

    @assert variable != "beta^2"
    y_vals = sort(unique([eval(Meta.parse(gp))[2] for gp in gridpoints]))
    displayed_y_vals = variable == "a_b" ? y_vals : sqrt.(y_vals)

    # Create matrix for heatmap
    z = zeros(length(x_vals), length(y_vals))
    z_text = fill("", length(x_vals), length(y_vals))
    
    # Fill matrix
    for (i, x) in enumerate(x_vals)
        for (j, y) in enumerate(y_vals)
            gp = "[$x, $y]"
            row = df[(df.Source .== source) .& 
                    (df.Fixed .== fixed) .& 
                    (df.Variable .== variable) .& 
                    (df.Factor .== factor) .& 
                    (df.Gridpoint .== gp), :]
            if !isempty(row)
                z[i,j] = row[1, metric]
                z_text[i,j] = smart_round(z[i,j])
            end
        end
    end
    
    target = replace(fixed, "targets_" => "")
    # capizalize factor
    factor = uppercase(factor)

    label_variable = "a and b"
    if variable == "X" || variable == "Y" || variable == "Z"
        label_variable = L"m_{%$variable \to f}"
    end

    if occursin("Prediction", source)
        title = "Predicted " * L"P_{%$target, %$factor} " * " with varying $label_variable"
    elseif occursin("Sampled", source)
        title = "Sampled " * L"P_{%$target, %$factor} " * " with varying $label_variable"
    end
    
    p = heatmap(
        displayed_y_vals, 
        x_vals, 
        z,
        size=(1600, 1200),
        annotationsfontsize=8,
        xlabel=variable == "a_b" ? "b" : "σ",
        ylabel=variable == "a_b" ? "a" : "μ",
        title=title, #"$metric\n$source, $fixed, $variable, $factor",
        color=:viridis,
        margin=10mm,
        xguidefontsize=16,
        yguidefontsize=16,
        xtickfontsize=14,
        ytickfontsize=14,
        titlefontsize=16,
        legendfontsize=12,
    )
    
    # Add text annotations
    for i in 1:size(z,1), j in 1:size(z,2)
        if z_text[i,j] != ""
            annotate!([(displayed_y_vals[j], x_vals[i], text(z_text[i,j], :white, :center, 8))])
        end
    end
    
    return p
end

function plot_1D_variable(df, source, fixed, variable, factor, metric)
    filtered_df = filter(row -> row.Source == source && row.Fixed == fixed && row.Variable == variable && row.Factor == factor, df)

    # print filter configuration
    println("Source: $source, Fixed: $fixed, Variable: $variable, Factor: $factor, Metric: $metric")

    xs = filtered_df[:, "Gridpoint"]
    ys = filtered_df[:, metric]
    target = replace(fixed, "targets_" => "")
    # capizalize factor
    factor = uppercase(factor)

    if occursin("Prediction", source)
        label = "Predicted " * L"P_{%$target, %$factor}"
    elseif occursin("Sampled", source)
        label = "Sampled " * L"P_{%$target, %$factor}"
    end

    return xs, ys, label, source, fixed, variable, factor, metric
end

function plot_metrics_fixed(datapath, savedir)
    mkpath(savedir)
    data = CSV.read(datapath, DataFrame)

    metrics = ["MAE_Marginal_Mean_mean","MAPE_Marginal_Mean_mean", "MAPE_Marginal_Variance_mean"]

    plots_beta2_variances = []
    plots_beta2_means = []

    for factor in factors
        combinations = get_valid_combinations(factor)
        for source in sources
            for (fixed, variable) in combinations
                if variable == "beta^2"
                    for metric in metrics
                        p = plot_1D_variable(data, source, fixed, variable, factor, metric)
                        push!(metric == "MAPE_Marginal_Variance_mean" ? plots_beta2_variances : plots_beta2_means, p)
                    end
                else
                    for metric in metrics
                        p = plot_2D_variable(data, source, fixed, variable, factor, metric)
                        filename = "$(source)_$(fixed)_$(variable)_$(factor)_$(metric).svg"
                        savefig(p, joinpath(savedir, filename))
                    end
                end
            end
        end
    end

    # Save plots for beta^2
    aggregated_plot_beta2_means = plot(
        plots_beta2_means[1][1], 
        [p[2] for p in plots_beta2_means], 
        label=reshape([p[3] for p in plots_beta2_means], (1, length(plots_beta2_means))), 
        size=(800, 600),
        xlabel=L"\beta^2",
        ylabel="MAPE",
        title="MAPE Marginal Mean vs. Exact",
        xguidefontsize=16,
        yguidefontsize=16,
        xtickfontsize=14,
        ytickfontsize=14,
        titlefontsize=16,
        legendfontsize=12,
    )
    aggregated_plot_beta2_variances = plot(
        plots_beta2_variances[1][1], 
        [p[2] for p in plots_beta2_variances], 
        label=reshape([p[3] for p in plots_beta2_variances], (1, length(plots_beta2_variances))), 
        size=(800, 600),
        xlabel=L"\beta^2",
        ylabel="MAPE",
        title="MAPE Marginal Variance vs. Exact",
        xguidefontsize=16,
        yguidefontsize=16,
        xtickfontsize=14,
        ytickfontsize=14,
        titlefontsize=16,
        legendfontsize=12,
    )

    combined_plot = plot(aggregated_plot_beta2_means, aggregated_plot_beta2_variances, layout=(1, 2), size=(1600, 600), margin=10mm)
    savefig(combined_plot, joinpath(savedir, "beta2_combined.svg"))
end

datapath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_fixed_large.csv"
savedir = "SoHEstimation/approximate_message_passing/evaluation/plots/plots_1fixed"

plot_metrics_fixed(datapath, savedir)