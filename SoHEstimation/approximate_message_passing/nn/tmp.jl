include("../nn/mjl.jl")
using .NN: get_data, remove_outliers
using Plots
using LaTeXStrings
using Measures

datapath = "SoHEstimation/approximate_message_passing/data_masterarbeit/dataset_em_factor_10_000_1000000_Experiment_1.jld2"
global plot_once = false

label_map = Dict(
	1 => "Mean of I",
	2 => "Standard deviation of I",
	3 => "Mean of SoH",
	4 => "Standard deviation of SoH",
	5 => "Mean of " * L"\Delta SoC",
	6 => "Standard deviation of " * L"\Delta SoC",
	7 => "Mean of " * L"\Delta q",
	8 => "Mean of " * L"\Delta t",
)

#model = load_model()

for target in ["targets_X", "targets_Y", "targets_Z"]
    factor = :emf
    X, y = get_data(datapath; target = target, factor = factor, transform_to_tau_rho = false, verbose = 0)
    X, _, _, _ = remove_outliers(X, y; verbose = 0)

    # Plot the distribution of each feature using scatter plot
	if !plot_once
		p = plot(layout = (4, 2), size = (2000, 2000), left_margin = 10mm, bottom_margin = 10mm)
		for i in 1:size(X, 2)
			if i < 7
				y_label =  L"\sigma"
				data_to_plot = (i % 2 == 0) ? sqrt.(X[:, i]) : X[:, i]
			else
				y_label =  L"\mu"
				data_to_plot = X[:, i]
			end
			scatter!(p[i], data_to_plot, alpha=0.75,
				xguidefontsize = 16, yguidefontsize = 16, xtickfontsize = 14, ytickfontsize = 14,
				titlefontsize = 18, legendfontsize = 12, legend = false)
			xlabel!(p[i], "Index")
			ylabel!(p[i], y_label)
			title!(p[i], "$(label_map[i])")
		end
		savefig(p, "SoHEstimation/approximate_message_passing/nn/figures/Scatter_Plot_Features.png")
		global plot_once = true
	end

	# Now plot y with (n,2)
	p_target = plot(layout = (1, 2), size = (1200, 400), left_margin = 10mm, right_margin = 10mm, top_margin = 10mm, bottom_margin = 10mm)
	target_label = replace(target, "targets_" => "")
	for i in 1:2
		data_to_plot = (i % 2 == 0) ? sqrt.(y[:, i]) : y[:, i]
		scatter!(p_target[i], data_to_plot, alpha=0.75,
			xguidefontsize = 16, yguidefontsize = 16, xtickfontsize = 14, ytickfontsize = 14,
			titlefontsize = 16, legendfontsize = 12, legend = false)
		xlabel!(p_target[i], "Index")
		
		if i == 1
			ylabel!(p_target[i], L"\mu")
			title!(p_target[i], "Mean of " * L"P_{%$target_label, EMF}")
		else
			ylabel!(p_target[i], L"\sigma")
			title!(p_target[i], "Standard deviation of " * L"P_{%$target_label, EMF}")
		end
	end
	savefig(p_target, "SoHEstimation/approximate_message_passing/nn/figures/Scatter_Plot_Targets_$(target_label).png")

	
end
