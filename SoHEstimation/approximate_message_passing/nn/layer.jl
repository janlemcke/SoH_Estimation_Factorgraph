module Layer

export ResidualMinimum, SoftplusLayer, ReluLayer, ClampLayer

using Flux: @functor, Chain, Dense
using Flux
using LogExpFunctions
using MLJBase
using CUDA
using Zygote

# this layer is used to enforce that the precision of the new marginal is larger than the precision of the incoming message
struct ResidualMinimum
	scale::Symbol
	scaling_params::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing}
	scale_output::Symbol
	scaling_params_output::Union{Tuple{Matrix{Float32}, Matrix{Float32}}, Nothing}
	moment::Symbol
	model::Chain
	transform_to_tau_rho::Bool
	input_index::Int
	output_index::Int
	positivity_constraint::String
end
function (l::ResidualMinimum)(x::AbstractMatrix)
	output = l.model(x)

	epsilon = 1e-2

	# check if type of x is x: Matrix{Float32}
	if typeof(x) == Matrix{Float32} && CUDA.functional()
		x = CuArray(x)
	end

	if typeof(output) == Matrix{Float32} && CUDA.functional()
		output = CuArray(output)
	end

	if l.scale == :minmax
		# Undo Min-Max scaling
		min_vals, max_vals = l.scaling_params
		if CUDA.functional()
			min_vals, max_vals = CuArray(min_vals), CuArray(max_vals)
		end
		range_vals = max_vals .- min_vals
		descaled_x = x .* range_vals' .+ min_vals'

	elseif l.scale == :zscore
		# Undo Z-score normalization
		means, stds = l.scaling_params
		if CUDA.functional()
			means = CuArray(means)
			stds = CuArray(stds)
		end
		descaled_x = x .* stds' .+ means'
	end

	if size(x, 1) == 9 # WSF
		if l.input_index == 2
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 1, 0, 0, 0, 0, 0, 0, 0]') : [0, 1, 0, 0, 0, 0, 0, 0, 0]'
		elseif l.input_index == 4
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 0, 0, 1, 0, 0, 0, 0, 0]') : [0, 0, 0, 1, 0, 0, 0, 0, 0]'
		elseif l.input_index == 6
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 0, 0, 0, 0, 1, 0, 0, 0]') : [0, 0, 0, 0, 0, 1, 0, 0, 0]'
		else
			error("Input index not recognized!")
		end
	elseif size(x, 1) == 5 # GMF
		if l.input_index == 2
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 1, 0, 0, 0]') : [0, 1, 0, 0, 0]'
		elseif l.input_index == 4
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 0, 0, 1, 0]') : [0, 0, 0, 1, 0]'
		end
	elseif size(x, 1) == 8 # EMF
		if l.input_index == 2
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 1, 0, 0, 0, 0, 0, 0]') : [0, 1, 0, 0, 0, 0, 0, 0]'
		elseif l.input_index == 4
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 0, 0, 1, 0, 0, 0, 0]') : [0, 0, 0, 1, 0, 0, 0, 0]'
		elseif l.input_index == 6
			selection_input_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 0, 0, 0, 0, 1, 0, 0]') : [0, 0, 0, 0, 0, 1, 0, 0]'
		else
			error("Input index not recognized!")
		end
	end

	selection_second_moment_output_vector = Zygote.@ignore CUDA.functional() ? CuArray([0, 1]') : [0, 1]'
	selection_first_moment_output_vector = Zygote.@ignore CUDA.functional() ? CuArray([1, 0]') : [1, 0]'

	if l.moment == :both
		if l.scale_output == :minmax
			# Undo Min-Max scaling
			min_vals_output, max_vals_output = l.scaling_params_output
			if CUDA.functional()
				min_vals_output, max_vals_output = CuArray(min_vals_output), CuArray(max_vals_output)
			end
			range_vals_output = max_vals_output .- min_vals_output
			descaled_output = output .* range_vals_output' .+ min_vals_output'
		elseif l.scale_output == :zscore
			# Undo Z-score normalization
			means_output, stds_output = l.scaling_params_output
			if CUDA.functional()
				means_output = CuArray(means_output)
				stds_output = CuArray(stds_output)
			end
			descaled_output = output .* stds_output' .+ means_output'
		end

		if l.scale != :none
			if !l.transform_to_tau_rho
				if l.scale_output == :none
					output_first_feature = selection_first_moment_output_vector * output
					output_second_feature = selection_second_moment_output_vector * output

					if l.positivity_constraint == "relu"
						output_second_feature = max.(1e-8, output_second_feature)
					elseif l.positivity_constraint == "softplus"
						output_second_feature = LogExpFunctions.log1pexp.(output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * descaled_x

					diff = max.(epsilon, input_second_moment_feature - output_second_feature - max.(epsilon, input_second_moment_feature .* (epsilon)))

					return vcat(output_first_feature, diff)
					#return [i[1] == l.oudtput_index ? max(0, descaled_x[l.input_index, i[2]] - output[i] .- epsilon) : output[i] for i in CartesianIndices(output)]
				else
					# output = [i[1] == l.output_index ? max(0, descaled_x[l.input_index, i[2]] - descaled_output[i] .- epsilon) : descaled_output[i] for i in CartesianIndices(descaled_output)]
					descaled_output_first_feature = selection_first_moment_output_vector * descaled_output
					descaled_output_second_feature = selection_second_moment_output_vector * descaled_output

					if l.positivity_constraint == "relu"
						descaled_output_second_feature = max.(1e-8, descaled_output_second_feature)
					elseif l.positivity_constraint == "softplus"
						descaled_output_second_feature = LogExpFunctions.log1pexp.(descaled_output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * descaled_x

					diff = max.(epsilon, input_second_moment_feature - descaled_output_second_feature - max.(epsilon, input_second_moment_feature .* (epsilon)))
					# println("input_second_moment_feature: $input_second_moment_feature descaled_output_second_feature: $descaled_output_second_feature maxepsilon: $(max.(epsilon, input_second_moment_feature .* (1 - epsilon)))")

					rescaled_output = vcat(descaled_output_first_feature, diff)
					# println("unscaled_input (means_output=$(l.scaling_params[1]), stds_output=$(l.scaling_params[2])):")
					# println(descaled_x)
					# println("unscaled_output (means_output=$means_output, stds_output=$stds_output):")
					# println(rescaled_output)
					if l.scale_output == :minmax
						rescaled_output = (rescaled_output .- min_vals_output') ./ range_vals_output'
					elseif l.scale_output == :zscore
						rescaled_output = (rescaled_output .- means_output') ./ stds_output'
					end
					return rescaled_output
				end
			else
				if l.scale_output == :none
					output_first_feature = selection_first_moment_output_vector * output
					output_second_feature = selection_second_moment_output_vector * output

					if l.positivity_constraint == "relu"
						output_second_feature = max.(1e-8, output_second_feature)
					elseif l.positivity_constraint == "softplus"
						output_second_feature = LogExpFunctions.log1pexp.(output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * descaled_x

					diff = input_second_moment_feature .+ output_second_feature .+ epsilon

					return vcat(output_first_feature, diff)
					# return [i[1] == l.output_index ? output[i] + epsilon + descaled_x[l.input_index, i[2]] .+ epsilon : output[i] for i in CartesianIndices(output)]
				else
					descaled_output_first_feature = selection_first_moment_output_vector * descaled_output
					descaled_output_second_feature = selection_second_moment_output_vector * descaled_output

					if l.positivity_constraint == "relu"
						descaled_output_second_feature = max.(1e-8, descaled_output_second_feature)
					elseif l.positivity_constraint == "softplus"
						descaled_output_second_feature = LogExpFunctions.log1pexp.(descaled_output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * descaled_x

					diff = input_second_moment_feature .+ descaled_output_second_feature .+ epsilon

					rescaled_output = vcat(descaled_output_first_feature, diff)
					if l.scale_output == :minmax
						rescaled_output = (rescaled_output .- min_vals_output') ./ range_vals_output'
					elseif l.scale_output == :zscore
						rescaled_output = (rescaled_output .- means_output') ./ stds_output'
					end
					return rescaled_output
				end

			end
		else
			if !l.transform_to_tau_rho
				if l.scale_output == :none
					output_first_feature = selection_first_moment_output_vector * output
					output_second_feature = selection_second_moment_output_vector * output

					if l.positivity_constraint == "relu"
						output_second_feature = max.(1e-8, output_second_feature)
					elseif l.positivity_constraint == "softplus"
						output_second_feature = LogExpFunctions.log1pexp.(output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * x

					diff = max.(epsilon, input_second_moment_feature - output_second_feature - max.(epsilon, input_second_moment_feature .* (epsilon)))

					return vcat(output_first_feature, diff)
					#return [i[1] == l.output_index ? max(0, x[l.input_index, i[2]] - output[i] .- epsilon) : output[i] for i in CartesianIndices(output)]
				else
					# output = [i[1] == l.output_index ? max(0, x[l.input_index, i[2]] - descaled_output[i] .- epsilon) : descaled_output[i] for i in CartesianIndices(descaled_output)]
					descaled_output_first_feature = selection_first_moment_output_vector * descaled_output
					descaled_output_second_feature = selection_second_moment_output_vector * descaled_output

					if l.positivity_constraint == "relu"
						descaled_output_second_feature = max.(1e-8, descaled_output_second_feature)
					elseif l.ositivity_constraint == "softplus"
						descaled_output_second_feature = LogExpFunctions.log1pexp.(descaled_output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * x

					diff = max.(epsilon, input_second_moment_feature - descaled_output_second_feature - max.(epsilon, input_second_moment_feature .* (epsilon)))

					rescaled_output = vcat(descaled_output_first_feature, diff)
					if l.scale_output == :minmax
						rescaled_output = (rescaled_output .- min_vals_output') ./ range_vals_output'
					elseif l.scale_output == :zscore
						rescaled_output = (rescaled_output .- means_output') ./ stds_output'
					end
					return rescaled_output
				end
			else
				if l.scale_output == :none
					output_first_feature = selection_first_moment_output_vector * output
					output_second_feature = selection_second_moment_output_vector * output

					if l.positivity_constraint == "relu"
						output_second_feature = max.(1e-8, output_second_feature)
					elseif l.positivity_constraint == "softplus"
						output_second_feature = LogExpFunctions.log1pexp.(output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * x

					diff = input_second_moment_feature .+ output_second_feature .+ epsilon

					return vcat(output_first_feature, diff)
					#return [i[1] == l.output_index ? output[i] + x[l.input_index, i[2]] .+ epsilon  : output[i] for i in CartesianIndices(output)]
				else
					#output = [i[1] == l.output_index ? descaled_output[i] + x[l.input_index, i[2]] .+ epsilon : descaled_output[i] for i in CartesianIndices(descaled_output)]
					descaled_output_first_feature = selection_first_moment_output_vector * descaled_output
					descaled_output_second_feature = selection_second_moment_output_vector * descaled_output

					if l.positivity_constraint == "relu"
						descaled_output_second_feature = max.(1e-8, descaled_output_second_feature)
					elseif l.positivity_constraint == "softplus"
						descaled_output_second_feature = LogExpFunctions.log1pexp.(descaled_output_second_feature)
					end

					input_second_moment_feature = selection_input_vector * x

					diff = input_second_moment_feature .+ descaled_output_second_feature .+ epsilon

					rescaled_output = vcat(descaled_output_first_feature, diff)
					if l.scale_output == :minmax
						rescaled_output = (rescaled_output .- min_vals_output') ./ range_vals_output'
					elseif l.scale_output == :zscore
						rescaled_output = (rescaled_output .- means_output') ./ stds_output'
					end
					return rescaled_output
				end
			end
		end

	elseif l.moment == :second
		if l.scale != :none
			if !l.transform_to_tau_rho
				return max.(1e-8, descaled_x[l.input_index, :] .- output' .- epsilon)
			else
				return output .+ descaled_x[l.input_index, :]' .+ epsilon
			end
		else
			if !l.transform_to_tau_rho
				return max.(1e-8, x[l.input_index, :] .- output' .- epsilon)
			else
				return output .+ x[l.input_index, :]' .+ epsilon
			end
		end
	else
		return output
	end
end
Flux.@functor ResidualMinimum
Flux.trainable(layer::ResidualMinimum) = (; model = layer.model)

end