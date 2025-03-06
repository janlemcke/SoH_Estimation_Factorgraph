using DelimitedFiles
using Plots
using Statistics
using CSV
using DataFrames

# Define the parameters
factors = ["wsf", "gmf"]
samples = ["10000000", "1000000", "100000", "10000", "1000"]
ns = ["20000", "10000", "5000", "2000", "1000", "500", "200", "100"]
wsf_targets = ["X", "Y", "Z"]
gmf_targets = ["X", "Y"]

# Function to filter data for specific parameters
function filter_data(df, source, factor, n, sample, target)
    filtered = df[(df.Source .== source) .&
                  (df.Factor .== factor) .& 
                  (df.N .== parse(Int, n)) .& 
                  (df.SamplesPerInput .== parse(Int, sample)) .&
                  (df.Target .== "targets_$target"), :]
    return filtered
end

# Function to compute first source data (Sampled_vs_Exact)
function compute_first_source_data(df, factor, n, sample, metric, aggregate=true)
    targets = factor == "wsf" ? wsf_targets : gmf_targets
    values = Float64[]
    
    for target in targets
        filtered_data = filter_data(df, "Sampled_vs_Exact", factor, n, sample, target)
        if size(filtered_data, 1) > 0
            value = filtered_data[1, metric]
            push!(values, value)
        end
    end
    
    if aggregate
        return !isempty(values) ? [mean(values)] : Float64[]
    else
        return values
    end
end

# Function to process second source data (Prediction_vs_Exact)
function process_second_source_data(df)
    results = Dict()
    for factor in factors
        for target in (factor == "wsf" ? wsf_targets : gmf_targets)
            results["$(factor)_$(target)"] = Dict(
                "mae_mean_ndata" => Dict{String, Float64}(),
                "mape_var_ndata" => Dict{String, Float64}(),
                "mae_mean_nsamples" => Dict{String, Float64}(),
                "mape_var_nsamples" => Dict{String, Float64}()
            )
        end
    end
    
    # Process data for each factor, n, and target combination
    for factor in factors
        targets = factor == "wsf" ? wsf_targets : gmf_targets
        for target in targets
            # Process ndata relationships (fixed sample size = 1000000)
            for n in ns
                filtered = filter_data(df, "Prediction_vs_Exact", factor, n, "1000000", target)
                if size(filtered, 1) > 0
                    results["$(factor)_$(target)"]["mae_mean_ndata"][n] = filtered[1, :MAE_Marginal_Mean]
                    results["$(factor)_$(target)"]["mape_var_ndata"][n] = filtered[1, :MAPE_Marginal_Variance]
                end
            end
            
            # Process nsamples relationships (fixed n = 5000)
            for sample in samples
                filtered = filter_data(df, "Prediction_vs_Exact", factor, "5000", sample, target)
                if size(filtered, 1) > 0
                    results["$(factor)_$(target)"]["mae_mean_nsamples"][sample] = filtered[1, :MAE_Marginal_Mean]
                    results["$(factor)_$(target)"]["mape_var_nsamples"][sample] = filtered[1, :MAPE_Marginal_Variance]
                end
            end
        end
    end
    
    return results
end

function create_plots(first_source_data, second_source_data, aggregate=true)
    plot_types = ["mae_mean_ndata", "mape_var_ndata", "mae_mean_nsamples", "mape_var_nsamples"]
    
    # Convert string arrays to numeric for plotting
    ns_numeric = parse.(Float64, ns)
    samples_numeric = parse.(Float64, samples)
    
    # Define color schemes
    colors = [:blue, :red, :green, :purple, :orange, :brown]
    
    for factor in factors
        targets = factor == "wsf" ? wsf_targets : gmf_targets
        
        for plot_type in plot_types
            p1 = plot(
                title="$(factor) $(plot_type)", 
                legend=:outertopright,
                xscale=:log10
            )
            
            # Set x-axis values based on plot type
            x_values = if contains(plot_type, "_ndata")
                ns_numeric
            else
                samples_numeric
            end
            
            xlabel!(contains(plot_type, "_ndata") ? "# data points" : "# samples per data point")
            
            # Plot first source data (sampled) - with dashed lines
            if aggregate
                data = first_source_data["$(factor)_$(plot_type)"]
                plot!(p1, x_values, data, 
                      label="sampled", 
                      linestyle=:dash, 
                      color=:blue,
                      marker=:circle)
            else
                for (i, target) in enumerate(targets)
                    data = first_source_data["$(factor)_$(target)_$(plot_type)"]
                    plot!(p1, x_values, data, 
                          label="sampled_$(target)", 
                          linestyle=:dash, 
                          color=colors[i],
                          marker=:circle)
                end
            end
            
            # Plot second source data (nn) - with solid lines
            if aggregate
                data = []
                for target in targets
                    target_data = collect(values(second_source_data["$(factor)_$(target)"][plot_type]))
                    push!(data, target_data)
                end
                plot!(p1, x_values, mean(data, dims=1)[:], 
                      label="nn", 
                      linestyle=:solid, 
                      color=:blue,
                      marker=:square)
            else
                for (i, target) in enumerate(targets)
                    data = collect(values(second_source_data["$(factor)_$(target)"][plot_type]))
                    plot!(p1, x_values, data, 
                          label="nn_$(target)", 
                          linestyle=:solid, 
                          color=colors[i],
                          marker=:square)
                end
            end
            
            # Save plot
            path = "SoHEstimation/approximate_message_passing/evaluation/results/nsamples_vs_ndata_plots_$(aggregate ? "aggregated" : "single")/$(factor)_$(plot_type).png"
            mkpath(dirname(path))
            savefig(p1, path)
        end
    end
end

# Main execution
function main(csv_path, aggregate=true)
    # Read the CSV file
    df = CSV.read(csv_path, DataFrame)
    
    # Process first source
    first_source_data = Dict()
    for factor in factors
        for plot_type in ["mae_mean_ndata", "mape_var_ndata", "mae_mean_nsamples", "mape_var_nsamples"]
            if aggregate
                first_source_data["$(factor)_$(plot_type)"] = []
            else
                for target in (factor == "wsf" ? wsf_targets : gmf_targets)
                    first_source_data["$(factor)_$(target)_$(plot_type)"] = []
                end
            end
        end
    end
    
    # Fill first source data
    for factor in factors
        for n in ns, sample in samples
            if sample == "1000000"
                data_mae = compute_first_source_data(df, factor, n, sample, :MAE_Marginal_Mean, aggregate)
                data_mape = compute_first_source_data(df, factor, n, sample, :MAPE_Marginal_Variance, aggregate)
                
                if aggregate
                    push!(first_source_data["$(factor)_mae_mean_ndata"], data_mae[1])
                    push!(first_source_data["$(factor)_mape_var_ndata"], data_mape[1])
                else
                    for (i, target) in enumerate((factor == "wsf" ? wsf_targets : gmf_targets))
                        push!(first_source_data["$(factor)_$(target)_mae_mean_ndata"], data_mae[i])
                        push!(first_source_data["$(factor)_$(target)_mape_var_ndata"], data_mape[i])
                    end
                end
            end
            
            if n == "5000"
                data_mae = compute_first_source_data(df, factor, n, sample, :MAE_Marginal_Mean, aggregate)
                data_mape = compute_first_source_data(df, factor, n, sample, :MAPE_Marginal_Variance, aggregate)
                
                if aggregate
                    push!(first_source_data["$(factor)_mae_mean_nsamples"], data_mae[1])
                    push!(first_source_data["$(factor)_mape_var_nsamples"], data_mape[1])
                else
                    for (i, target) in enumerate((factor == "wsf" ? wsf_targets : gmf_targets))
                        push!(first_source_data["$(factor)_$(target)_mae_mean_nsamples"], data_mae[i])
                        push!(first_source_data["$(factor)_$(target)_mape_var_nsamples"], data_mape[i])
                    end
                end
            end
        end
    end
    
    # Process second source
    second_source_data = process_second_source_data(df)
    
    # Create plots
    create_plots(first_source_data, second_source_data, aggregate)
end

# Run the script with desired aggregation setting
main("SoHEstimation/approximate_message_passing/evaluation/results/metrics_marginal_vs_exact.csv", true)  # Set second parameter to false for no aggregation