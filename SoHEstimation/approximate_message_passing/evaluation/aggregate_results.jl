using CSV
using DataFrames
using StatsBase

function compute_aggregation_across_experiments()
    filepath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_marginal_vs_exact_MA.csv"
    df = CSV.read(filepath, DataFrame)

    # Define metrics to aggregate
    metrics = ["RMSE_Marginal_Mean", "RMSE_Marginal_Variance", "RMSE_Msg_Mean", "RMSE_Msg_Variance",
               "MAE_Marginal_Mean", "MAE_Marginal_Variance", "MAE_Msg_Mean", "MAE_Msg_Variance",
               "MAPE_Marginal_Mean", "MAPE_Marginal_Variance", "MAPE_Msg_Mean", "MAPE_Msg_Variance"]

    # Extract unique Source-Factor-Target-SamplesPerInput pairs
    unique_groups = unique(select(df, [:Source, :Factor, :Target, :SamplesPerInput]))

    # Create an empty DataFrame for aggregated results
    agg_df = DataFrame(Source=String[], Factor=String[], Target=String[], SamplesPerInput=Int[], N=Int[])

    # Add separate columns for mean and std dynamically
    for metric in metrics
        agg_df[!, metric * "_Mean"] = Float64[]
        agg_df[!, metric * "_Std"] = Float64[]
    end

    # Process each unique group
    for row in eachrow(unique_groups)
        source = row[:Source]
        factor = row[:Factor]
        target = row[:Target]
        samples_per_input = row[:SamplesPerInput]

        # Filter and sort data
        subdf = filter(r -> r[:Source] == source && r[:Factor] == factor && r[:Target] == target && r[:SamplesPerInput] == samples_per_input, df)
        subdf = sort(subdf, :N)

        for n_val in unique(subdf.N)
            n_subdf = filter(r -> r[:N] == n_val, subdf)

            # Prepare row for aggregated results
            new_row = Dict(:Source => source, :Factor => factor, :Target => target, :SamplesPerInput => samples_per_input, :N => n_val)

            for metric in metrics
                new_row[Symbol(metric * "_Mean")] = StatsBase.mean(n_subdf[:, metric])
                new_row[Symbol(metric * "_Std")] = StatsBase.std(n_subdf[:, metric])
            end

            push!(agg_df, new_row)
        end
    end

    # Save aggregated results to CSV
    output_filepath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_marginal_vs_exact_aggregated_MA.csv"
    CSV.write(output_filepath, agg_df)

    println("Aggregated metrics saved to '$output_filepath'")
end

compute_aggregation_across_experiments()
