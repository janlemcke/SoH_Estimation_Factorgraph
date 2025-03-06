using CSV
using DataFrames
using Plots, Measures
using LaTeXStrings

# Load aggregated data
filepath = "SoHEstimation/approximate_message_passing/evaluation/results/metrics_marginal_vs_exact_aggregated_MA.csv"
df = CSV.read(filepath, DataFrame)

# Create a directory to save plots
plot_dir = "SoHEstimation/approximate_message_passing/evaluation/plots"
if !isdir(plot_dir)
	mkdir(plot_dir)
end

function plot_n()
	# Extract unique Factors and Targets
	factors = unique(df.Factor)
	targets = unique(df.Target)

	# Loop through Factors and Targets, plotting everything in the same figure
	for source in ["Sampled_vs_Exact", "Prediction_vs_Exact"]
		# Initialize subplots with extra left margin
        p1 = plot(title = "RMSE Marginal Mean", xlabel = L"N", ylabel = "RMSE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
            yguidefontsize = 16,
            xtickfontsize = 14,
            ytickfontsize = 14,
            titlefontsize = 16,
            legendfontsize = 12, legend = :topleft)
        p2 = plot(title = "RMSE Marginal Variance", xlabel = L"N", ylabel = "RMSE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
            yguidefontsize = 16,
            xtickfontsize = 14,
            ytickfontsize = 14,
            titlefontsize = 16,
            legendfontsize = 12, legend = :topleft)
        p3 = plot(title = "MAPE Marginal Mean", xlabel = L"N", ylabel = "MAPE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
            yguidefontsize = 16,
            xtickfontsize = 14,
            ytickfontsize = 14,
            titlefontsize = 16,
            legendfontsize = 12, legend = :topleft)
        p4 = plot(title = "MAPE Marginal Variance", xlabel = L"N", ylabel = "MAPE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
            yguidefontsize = 16,
            xtickfontsize = 14,
            ytickfontsize = 14,
            titlefontsize = 16,
            legendfontsize = 12, legend = :topleft)
		for factor in factors
			for target in targets
				subdf = filter(r -> r.Factor == factor && r.Target == target && r.SamplesPerInput == 1_000_000 && r.Source == source, df)
				subdf = sort(subdf, :N)

				if nrow(subdf) > 0  # Ensure there's data to plot
                    target_label = replace(target, "targets_" => "")
					label_str = L"P_{%$target_label, %$factor}"
                    

					# Add lines to each subplot
					plot!(p1, subdf.N, subdf.RMSE_Marginal_Mean_Mean, yerror = subdf.RMSE_Marginal_Mean_Std, label = label_str)
					plot!(p2, subdf.N, subdf.RMSE_Marginal_Variance_Mean, yerror = subdf.RMSE_Marginal_Variance_Std, label = label_str)
					plot!(p3, subdf.N, subdf.MAPE_Marginal_Mean_Mean, yerror = subdf.MAPE_Marginal_Mean_Std, label = label_str)
					plot!(p4, subdf.N, subdf.MAPE_Marginal_Variance_Mean, yerror = subdf.MAPE_Marginal_Variance_Std, label = label_str)
				end
			end
		end

		# Combine all subplots into a single 2x2 grid
        final_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 1000), margin = 10mm)

		# Save the plot
		savefig(final_plot, plot_dir * "/Metrics_Comparison_AllFactors_$source.svg")
	end
end


function plot_nsamples()
	# Extract unique Factors and Targets
	factors = unique(df.Factor)
	targets = unique(df.Target)
	for source in ["Sampled_vs_Exact", "Prediction_vs_Exact"]

		# Initialize subplots with extra left margin
		p1 = plot(title = "RMSE Marginal Mean", xlabel = L"N_s", ylabel = "RMSE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
        yguidefontsize = 16,
        xtickfontsize = 14,
        ytickfontsize = 14,
        titlefontsize = 16,
        legendfontsize = 12)
		p2 = plot(title = "RMSE Marginal Variance", xlabel = L"N_s", ylabel = "RMSE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
        yguidefontsize = 16,
        xtickfontsize = 14,
        ytickfontsize = 14,
        titlefontsize = 16,
        legendfontsize = 12)
		p3 = plot(title = "MAPE Marginal Mean", xlabel = L"N_s", ylabel = "MAPE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
        yguidefontsize = 16,
        xtickfontsize = 14,
        ytickfontsize = 14,
        titlefontsize = 16,
        legendfontsize = 12)
		p4 = plot(title = "MAPE Marginal Variance", xlabel = L"N_s", ylabel = "MAPE", lw = 2, xscale = :log10, left_margin = 10mm, xguidefontsize = 16,
        yguidefontsize = 16,
        xtickfontsize = 14,
        ytickfontsize = 14,
        titlefontsize = 16,
        legendfontsize = 12)

		# Loop through Factors, Targets, and N values
		for factor in factors
			for target in targets
				for n in [5000, 10000]  # Iterate over N values
					subdf = filter(r -> r.Factor == factor && r.Target == target && r.N == n && r.Source == source, df)
					subdf = sort(subdf, :SamplesPerInput)

                    target_label = replace(target, "targets_" => "")

					if nrow(subdf) > 0  # Ensure there's data to plot
                        label_str = L"P_{%$target_label, %$factor}, N_s=%$n"

						# Add lines to each subplot
						plot!(p1, subdf.SamplesPerInput, subdf.RMSE_Marginal_Mean_Mean, yerror = subdf.RMSE_Marginal_Mean_Std, label = label_str)
						plot!(p2, subdf.SamplesPerInput, subdf.RMSE_Marginal_Variance_Mean, yerror = subdf.RMSE_Marginal_Variance_Std, label = label_str)
						plot!(p3, subdf.SamplesPerInput, subdf.MAPE_Marginal_Mean_Mean, yerror = subdf.MAPE_Marginal_Mean_Std, label = label_str)
						plot!(p4, subdf.SamplesPerInput, subdf.MAPE_Marginal_Variance_Mean, yerror = subdf.MAPE_Marginal_Variance_Std, label = label_str)
					end
				end
			end
		end

		# Combine all subplots into a single 2x2 grid
		final_plot = plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 1000), margin = 10mm)

		# Save the plot
		savefig(final_plot, plot_dir * "/Metrics_Comparison_AllFactors_SamplesPerInput_$source.svg")
	end
end


# Run plotting functions
plot_n()
plot_nsamples()

println("Plots saved for all Factors combined in the '$plot_dir' directory!")
