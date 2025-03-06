using DelimitedFiles
using Plots
using Printf

function get_path_with_unique_filename(dir, base_filename, extension="pdf")
    counter = 0
    while true
        filename = "$(base_filename)_$(counter).$(extension)"
        full_path = joinpath(dir, filename)
        !isfile(full_path) && return full_path
        counter += 1
    end
end

# Define the parameters
factors = ["wsf", "gmf"]
n_samples_per_inputs = [10000000, 1000000, 100000, 10000, 1000]
n_datapoints = [10_000, 2_000, 1_000, 500, 200, 100]
targets_by_factor = Dict(
    "wsf" => ["X", "Y", "Z"],
    "gmf" => ["X", "Y"]
)

# Create output directory if it doesn't exist
output_dir = "SoHEstimation/approximate_message_passing/evaluation/results/plots_datasize_marginal_vs_exact"
mkpath(output_dir)

# Function to extract specific metrics from results
function extract_metrics(data)
    metrics = Dict()
    for (metric, value) in eachrow(data)
        metrics[metric] = value
    end
    return metrics
end

# Colors for different targets
target_colors = Dict(
    "X" => :blue,
    "Y" => :red,
    "Z" => :green
)

# Plot varying n_datapoints (fixed n_samples_per_input = 1000000)
for factor in factors
    plt = plot(
        title="MAPE vs Number of Datapoints ($factor)",
        xlabel="Number of Datapoints",
        ylabel="MAPE",
        xscale=:log10,
        yscale=:log10,
        legend=:outertopright
    )
    
    for target in targets_by_factor[factor]
        mape_marginal_mean = Float64[]
        mape_marginal_var = Float64[]
        mape_msg_mean = Float64[]
        mape_msg_var = Float64[]
        
        for n in n_datapoints
            path = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_$(factor)_$(n)_1000000_targets_$(target).txt"
            data = readdlm(path)
            metrics = extract_metrics(data)
            
            push!(mape_marginal_mean, metrics["mape_marginal_mean"])
            push!(mape_marginal_var, metrics["mape_marginal_variance"])
            push!(mape_msg_mean, metrics["mape_msg_mean"])
            push!(mape_msg_var, metrics["mape_msg_variance"])
        end
        
        plot!(plt, n_datapoints, mape_marginal_mean, 
            label="Marginal Mean ($target)", 
            color=target_colors[target], 
            linestyle=:solid)
        plot!(plt, n_datapoints, mape_marginal_var, 
            label="Marginal Variance ($target)", 
            color=target_colors[target], 
            linestyle=:dash)
        # plot!(plt, n_datapoints, mape_msg_mean, 
        #     label="Message Mean ($target)", 
        #     color=target_colors[target], 
        #     linestyle=:dot)
        # plot!(plt, n_datapoints, mape_msg_var, 
        #     label="Message Variance ($target)", 
        #     color=target_colors[target], 
        #     linestyle=:dashdot)
    end

    savefig(plt, get_path_with_unique_filename(output_dir, "mape_vs_datapoints_$(factor)"))
end

# # Plot varying n_samples_per_input (fixed n_datapoints = 5000)
# for factor in factors
#     plt = plot(
#         title="MAPE vs Number of Samples per Input ($factor)",
#         xlabel="Number of Samples per Input",
#         ylabel="MAPE",
#         xscale=:log10,
#         yscale=:log10,
#         legend=:outertopright
#     )
    
#     for target in targets_by_factor[factor]
#         mape_marginal_mean = Float64[]
#         mape_marginal_var = Float64[]
#         mape_msg_mean = Float64[]
#         mape_msg_var = Float64[]
        
#         for n in n_samples_per_inputs
#             path = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_$(factor)_5000_$(n)_targets_$(target).txt"
#             data = readdlm(path)
#             metrics = extract_metrics(data)
            
#             push!(mape_marginal_mean, metrics["mape_marginal_mean"])
#             push!(mape_marginal_var, metrics["mape_marginal_variance"])
#             push!(mape_msg_mean, metrics["mape_msg_mean"])
#             push!(mape_msg_var, metrics["mape_msg_variance"])
#         end
        
#         plot!(plt, n_samples_per_inputs, mape_marginal_mean, 
#             label="Marginal Mean ($target)", 
#             color=target_colors[target], 
#             linestyle=:solid)
#         plot!(plt, n_samples_per_inputs, mape_marginal_var, 
#             label="Marginal Variance ($target)", 
#             color=target_colors[target], 
#             linestyle=:dash)
#         plot!(plt, n_samples_per_inputs, mape_msg_mean, 
#             label="Message Mean ($target)", 
#             color=target_colors[target], 
#             linestyle=:dot)
#         plot!(plt, n_samples_per_inputs, mape_msg_var, 
#             label="Message Variance ($target)", 
#             color=target_colors[target], 
#             linestyle=:dashdot)
#     end
    
#     savefig(plt, get_path_with_unique_filename(output_dir, "mape_vs_samples_$(factor)"))
# end