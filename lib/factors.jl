# A set of types and functions for factors over gaussian variables
#
# Hasso-Plattner Institute

module Factors
using Distributions: Distributions
export GaussianFactor, GaussianMeanFactor, ScalingFactor, WeightedSumFactor, ApproximateWeightedSumFactor, ApproximateGaussianMeanFactor, GreaterThanFactor, update_msg_to_x!, update_msg_to_y!, update_msg_to_z!, normalize_log_factor!, normalize_log_var!,
	Factor, normalize!, ApproximateEMFactor

# # debug code begin
include("../SoHEstimation/approximate_message_passing/em_factor/generate_dataset_em_factor.jl")
using .ElectricalModelFactorGeneration: generate_output_em_factor
# include("../SoHEstimation/approximate_message_passing/weighted_sum_factor/plot_nn_results.jl")
# # debug code end

using ..DistributionCollections
using ..GaussianDistribution
using ..NN
using Flux
using JLD2
using StatsBase


abstract type Factor end


struct GaussianFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	prior::Gaussian1D
	msg_to_x::Int64
end

GaussianFactor(db::DistributionBag{Gaussian1D}, x::Int64, prior) = GaussianFactor(db, x, prior, add!(db))

function update_msg_to_x!(f::GaussianFactor)
	oldMarginal = f.db[f.x]
	newMarginal = oldMarginal / f.db[f.msg_to_x] * f.prior
	f.db[f.x] = newMarginal
	f.db[f.msg_to_x] = f.prior
	return (absdiff(oldMarginal, newMarginal))
end

function normalize_log_var!(f::GaussianFactor)
	logZ = logNormProduct(f.db[f.x], f.db[f.msg_to_x])
	f.db[f.x] *= f.db[f.msg_to_x]
	return logZ
end

function normalize_log_factor!(f::GaussianFactor)
	return 0
end


struct GaussianMeanFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	beta_squared::Float64
	msg_to_x::Int64
	msg_to_y::Int64
end

GaussianMeanFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, beta_squared) = GaussianMeanFactor(db, x, y, beta_squared, add!(db), add!(db))

function update_msg_to_x!(f::GaussianMeanFactor)
	msgBack = f.db[f.y] / f.db[f.msg_to_y]
	c = 1 / (1 + f.beta_squared * msgBack.rho)
	newMsg = Gaussian1D(msgBack.tau * c, msgBack.rho * c)

	oldMarginal = f.db[f.x]
	newMarginal = oldMarginal / f.db[f.msg_to_x] * newMsg
	f.db[f.x] = newMarginal
	f.db[f.msg_to_x] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

function update_marginal_x!(msgBackX, msgBackY, beta_squared = 4)
	c = 1 / (1 + beta_squared * msgBackY.rho)
	newMsg = Gaussian1D(msgBackY.tau * c, msgBackY.rho * c)

	return msgBackX * newMsg
end

function update_marginal_y!(msgBackX, msgBackY, beta_squared = 4)
	c = 1 / (1 + beta_squared * msgBack.rho)
	newMsg = Gaussian1D(msgBackX.tau * c, msgBackX.rho * c)

	return msgBackY * newMsg
end

function update_msg_to_y!(f::GaussianMeanFactor)
	# if f.db[f.x].rho < f.db[f.msg_to_x].rho
	#   f.db[f.x] = Gaussian1DFromMeanVariance(mean(f.db[f.x]), variance(f.db[f.msg_to_x]))
	# end
	msgBack = f.db[f.x] / f.db[f.msg_to_x]
	c = 1 / (1 + f.beta_squared * msgBack.rho)
	newMsg = Gaussian1D(msgBack.tau * c, msgBack.rho * c)

	oldMarginal = f.db[f.y]
	newMarginal = oldMarginal / f.db[f.msg_to_y] * newMsg
	f.db[f.y] = newMarginal
	f.db[f.msg_to_y] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

function normalize_log_var!(f::GaussianMeanFactor)
	logZ = logNormProduct(f.db[f.x], f.db[f.msg_to_x])
	logZ += logNormProduct(f.db[f.y], f.db[f.msg_to_y])
	f.db[f.x] *= f.db[f.msg_to_x]
	f.db[f.y] *= f.db[f.msg_to_y]
	return (logZ)
end

function normalize_log_factor!(f::GaussianMeanFactor)
	logZ = logNormRatio(f.db[f.y], f.db[f.msg_to_y])
	return (logZ)
end

struct ApproximateGaussianMeanFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	beta_squared::Float64
	msg_to_x::Int64
	msg_to_y::Int64
	models::Vector{Any}  # Store the loaded models (NNs) directly here
end

@enum Variable X Y

ApproximateGaussianMeanFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, beta_squared::Float64, model_names::Vector{String}) = BuildApproximateGaussianMeanFactor(db, x, y, beta_squared, model_names)

function BuildApproximateGaussianMeanFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, beta_squared::Float64, model_names::Vector{String})
	if length(model_names) != 2 && length(model_names) != 4
		throw(ArgumentError("Models array must have either 2 or 4 elements"))
	end

	# Load models once based on the provided paths
	# loaded_models = [JLD2.load("SoHEstimation/approximate_message_passing/gaussian_mean_factor/models/" * name) for name in model_names]

	return ApproximateGaussianMeanFactor(db, x, y, beta_squared, add!(db), add!(db), loaded_models)
end

function update_msg_to_x!(f::ApproximateGaussianMeanFactor)
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]

	input = [GaussianDistribution.mean(msgBackY), GaussianDistribution.variance(msgBackY), f.beta_squared]
	input = normalize_sample(input, f.models[1]["norms"])  # Use the norms from the first model

	if length(f.models) == 2
		# Use the first model for both mean and variance for X
		newMsgMoments = f.models[1]["model"](input)
	elseif length(f.models) == 4
		# Use separate models for mean and variance for X
		newMsgMean = f.models[1]["model"](input)
		newMsgVariance = f.models[2]["model"](input)
		newMsgMoments = [newMsgMean, newMsgVariance]
	end

	print("newMsgMoments: ", newMsgMoments)
	newMsg = Gaussian1DFromMeanVariance(newMsgMoments[1], newMsgMoments[2])
	newMarginal = newMsg * msgBackX

	oldMarginal = f.db[f.x]
	f.db[f.x] = newMarginal
	f.db[f.msg_to_x] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

function update_msg_to_y!(f::ApproximateGaussianMeanFactor)
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]

	input = [GaussianDistribution.mean(msgBackX), GaussianDistribution.variance(msgBackX), f.beta_squared]

	if length(f.models) == 2
		input = normalize_sample(input, f.models[2]["norms"])  # Use the norms from the second model
		# Use the second model for both mean and variance for Y
		newMsgMoments = f.models[2]["model"](input)
	elseif length(f.models) == 4
		input = normalize_sample(input, f.models[3]["norms"])  # Use the norms from the second model
		# Use separate models for mean and variance for Y
		newMsgMean = f.models[3]["model"](input)
		newMsgVariance = f.models[4]["model"](input)
		newMsgMoments = [newMsgMean, newMsgVariance]
	end

	newMsg = Gaussian1DFromMeanVariance(newMsgMoments[1], newMsgMoments[2])
	newMarginal = newMsg * msgBackY

	oldMarginal = f.db[f.y]
	f.db[f.y] = newMarginal
	f.db[f.msg_to_y] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end


struct ScalingFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	a::Float64
	b::Float64
	msg_to_x::Int64
	msg_to_y::Int64
end

ScalingFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, a) = ScalingFactor(db, x, y, a, 0.0, add!(db), add!(db))
ScalingFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, a, b) = ScalingFactor(db, x, y, a, b, add!(db), add!(db))

function update_msg_to_x!(f::ScalingFactor)
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]

	# newMsgX = Gaussian1DFromMeanVariance(
	#     GaussianDistribution.mean(msgBackY) / f.a,
	#     GaussianDistribution.variance(msgBackY) / (f.a * f.a),
	# )
	# newMsgX = Gaussian1D(
	#   msgBackY.tau * f.a + f.a^2 * f.b * msgBackY.rho,
	#   msgBackY.rho * (f.a * f.a),
	# )
	newMsgX = Gaussian1D(
		msgBackY.tau * f.a - f.a * f.b * msgBackY.rho,
		msgBackY.rho * (f.a * f.a),
	)

	oldMarginal = f.db[f.x]
	newMarginal = oldMarginal / f.db[f.msg_to_x] * newMsgX
	f.db[f.x] = newMarginal
	f.db[f.msg_to_x] = newMsgX
	return (absdiff(oldMarginal, newMarginal))
end

function update_msg_to_y!(f::ScalingFactor)
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]

	newMsgY = Gaussian1DFromMeanVariance(
		f.a * GaussianDistribution.mean(msgBackX) + f.b,
		f.a * f.a * GaussianDistribution.variance(msgBackX),
	)

	oldMarginal = f.db[f.y]
	newMarginal = oldMarginal / f.db[f.msg_to_y] * newMsgY
	f.db[f.y] = newMarginal
	f.db[f.msg_to_y] = newMsgY
	return (absdiff(oldMarginal, newMarginal))
end


struct WeightedSumFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	z::Int64
	a::Float64
	b::Float64
	c::Float64
	msg_to_x::Int64
	msg_to_y::Int64
	msg_to_z::Int64
end

WeightedSumFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, z::Int64, a, b) = WeightedSumFactor(db, x, y, z, a, b, 0.0, add!(db), add!(db), add!(db))
WeightedSumFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, z::Int64, a, b, c) = WeightedSumFactor(db, x, y, z, a, b, c, add!(db), add!(db), add!(db))

function update_msg_to_x!(f::WeightedSumFactor)
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]

	newMsgX = Gaussian1DFromMeanVariance(
		GaussianDistribution.mean(msgBackZ) / f.a - f.b / f.a * GaussianDistribution.mean(msgBackY) - f.c / f.a,
		GaussianDistribution.variance(msgBackZ) / (f.a * f.a) + f.b * f.b / (f.a * f.a) * GaussianDistribution.variance(msgBackY),
	)

	oldMarginal = f.db[f.x]
	newMarginal = oldMarginal / f.db[f.msg_to_x] * newMsgX
	f.db[f.x] = newMarginal
	f.db[f.msg_to_x] = newMsgX
	return (absdiff(oldMarginal, newMarginal))
end

function update_msg_to_y!(f::WeightedSumFactor)
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]

	newMsgY = Gaussian1DFromMeanVariance(
		GaussianDistribution.mean(msgBackZ) / f.b - f.a / f.b * GaussianDistribution.mean(msgBackX) - f.c / f.b,
		GaussianDistribution.variance(msgBackZ) / (f.b * f.b) + f.a * f.a / (f.b * f.b) * GaussianDistribution.variance(msgBackX),
	)

	oldMarginal = f.db[f.y]
	newMarginal = oldMarginal / f.db[f.msg_to_y] * newMsgY
	f.db[f.y] = newMarginal
	f.db[f.msg_to_y] = newMsgY
	return (absdiff(oldMarginal, newMarginal))
end

function update_msg_to_z!(f::WeightedSumFactor)
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]

	newMsgZ = Gaussian1DFromMeanVariance(
		f.a * GaussianDistribution.mean(msgBackX) + f.b * GaussianDistribution.mean(msgBackY) + f.c,
		f.a * f.a * GaussianDistribution.variance(msgBackX) + f.b * f.b * GaussianDistribution.variance(msgBackY),
	)

	oldMarginal = f.db[f.z]
	newMarginal = oldMarginal / f.db[f.msg_to_z] * newMsgZ
	f.db[f.z] = newMarginal
	f.db[f.msg_to_z] = newMsgZ
	return (absdiff(oldMarginal, newMarginal))
end

function normalize_log_var!(f::WeightedSumFactor)
	logZ = logNormProduct(f.db[f.x], f.db[f.msg_to_x])
	logZ += logNormProduct(f.db[f.y], f.db[f.msg_to_y])
	logZ += logNormProduct(f.db[f.z], f.db[f.msg_to_z])
	f.db[f.x] *= f.db[f.msg_to_x]
	f.db[f.y] *= f.db[f.msg_to_y]
	f.db[f.z] *= f.db[f.msg_to_z]
	return (logZ)
end

function normalize_log_factor!(f::WeightedSumFactor)
	logZ = logNormRatio(f.db[f.x], f.db[f.msg_to_x]) + logNormRatio(f.db[f.y], f.db[f.msg_to_y])
	return (logZ)
end

struct ApproximateWeightedSumFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	z::Int64
	a::Float64
	b::Float64
	c::Float64
	msg_to_x::Int64
	msg_to_y::Int64
	msg_to_z::Int64
end

#nn_wsf_marginal_x = load_model("SoHEstimation\\approximate_message_passing\\models\\fastrun_targets_X__both_ttt_based.jls")
# nn_wsf_marginal_y = load("/Users/leonhard.hennicke/Documents/phd/phy-ml/SoHEstimation/approximate_message_passing/weighted_sum_factor/models/model_for_Y_20_000_kl_0_min_std_ALL_VARS.jld2")
# nn_wsf_marginal_z = load("/Users/leonhard.hennicke/Documents/phd/phy-ml/SoHEstimation/approximate_message_passing/weighted_sum_factor/models/model_for_Z_20_000_kl_0_min_std_ALL_VARS.jld2")
# nn_wsf_marginal_x = load_model("/Users/leonhard.hennicke/Documents/phd/phy-ml/SoHEstimation/approximate_message_passing/weighted_sum_factor/models/fastrun_targets_X_test.jls")
# nn_wsf_marginal_x = load_model("/Users/leonhard.hennicke/Documents/phd/phy-ml/SoHEstimation/approximate_message_passing/weighted_sum_factor/models/fastrun_targets_X_ (1).jld2")


# nn_wsf_marginal_x_1st = load_model("SoHEstimation/approximate_message_passing/models/fastrun_targets_X__first_zscore_activation_tanh_fast_outputlayer_softplus.jls")
# nn_wsf_marginal_x_2nd = load_model("SoHEstimation/approximate_message_passing/models/fastrun_targets_X__second_zscore_activation_tanh_fast_outputlayer_softplus.jls")

# nn_wsf_marginal_y_1st = load_model("SoHEstimation/approximate_message_passing/models/fastrun_targets_Y__first_zscore_activation_tanh_fast_outputlayer_relu.jls")
# nn_wsf_marginal_y_2nd = load_model("SoHEstimation/approximate_message_passing/models/fastrun_targets_Y__second_zscore_activation_tanh_fast_outputlayer_softplus.jls")

# nn_wsf_marginal_z_1st = load_model("SoHEstimation/approximate_message_passing/models/fastrun_targets_Z__first_zscore_activation_tanh_fast_outputlayer_relu.jls")
# nn_wsf_marginal_z_2nd = load_model("SoHEstimation/approximate_message_passing/models/fastrun_targets_Z__second_zscore_activation_relu_outputlayer_relu.jls")


#nn_wsf_marginl_y = load_model("SoHEstimation\\approximate_message_passing\\models\\fastrun_targets_Y__both_ttt_based_RESIDUAL_MAX.jls")
#nn_wsf_marginal_z = load_model("SoHEstimation\\approximate_message_passing\\models\\fastrun_targets_Z__both_ttt_based_RESIDUAL_MAX.jls")


ApproximateWeightedSumFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, z::Int64, a, b) = ApproximateWeightedSumFactor(db, x, y, z, a, b, 0.0, add!(db), add!(db), add!(db))
ApproximateWeightedSumFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, z::Int64, a, b, c) = ApproximateWeightedSumFactor(db, x, y, z, a, b, c, add!(db), add!(db), add!(db))

function normalize_input(input, norm)
	@assert length(input) == length(norm)
	return [norm[idx][:var] > 0 ? (element - norm[idx][:mean]) / norm[idx][:var] : element - norm[idx][:mean] for (idx, element) in enumerate(input)]
end

function update_msg_to_x!(f::ApproximateWeightedSumFactor)
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]

	input = [msgBackX.tau, msgBackX.rho, msgBackY.tau, msgBackY.rho, msgBackZ.tau, msgBackZ.rho, f.a, f.b, f.c]
	new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_x_1st, input)[1], predict_sample(nn_wsf_marginal_x_2nd, input)[1]

	#new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_x, input)
	newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2])

	# Clamp variance of marginal to avoid negative rho in newMsg.
	# Convert to mean/var first, since, if we only clamp rho, tau will be distorted (because it is a function of both mean and var).
	# Use prevfloat to make sure the variance does not get slightly bigger due to numerical errors (i.e. inversion of floats not being an involution).
	"""
	new_approx_marginal_mean = GaussianDistribution.mean(newMarginal)
	new_approx_marginal_variance = clamp(GaussianDistribution.variance(newMarginal), 0.0, prevfloat(GaussianDistribution.variance(msgBackX)) * 0.991)
	newMarginal = Gaussian1DFromMeanVariance(new_approx_marginal_mean, new_approx_marginal_variance)
	"""
	# newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], clamp(new_approx_marginal_tau_rho[2], msgBackX.rho, Inf))

	# # debug code begin
	# analytical_new_marginal = update_marginal_x(to_mean_variance(input)...)
	# if abs(GaussianDistribution.mean(analytical_new_marginal) - GaussianDistribution.mean(newMarginal)) > 0.1*abs(GaussianDistribution.mean(analytical_new_marginal))
	#   println("====================================")
	#   println("mean discrepancy in marginal x: ", abs(GaussianDistribution.mean(analytical_new_marginal) - GaussianDistribution.mean(newMarginal)))
	#   println("analytical marginal mean, var, tau, rho: ", GaussianDistribution.mean(analytical_new_marginal), " ", GaussianDistribution.variance(analytical_new_marginal), " ", analytical_new_marginal.tau, " ", analytical_new_marginal.rho)
	#   println("clamped approximate marginal mean, var, tau, rho: ", GaussianDistribution.mean(newMarginal), " ", GaussianDistribution.variance(newMarginal), " ", newMarginal.tau, " ", newMarginal.rho)
	#   println("unclamped approximate marginal mean, var, tau, rho: ", new_approx_marginal_tau_rho[1] / new_approx_marginal_tau_rho[2], " ", 1 / new_approx_marginal_tau_rho[2], " ", new_approx_marginal_tau_rho[1], " ", new_approx_marginal_tau_rho[2])
	#   println("input tau, rho: ", input)
	#   println("input mean, var: ", to_mean_variance(input))
	#   println("====================================")
	# end
	# # debug code end
	if newMarginal.rho < msgBackX.rho
		error("newMarginal.rho < msgBackX.rho (newMarginal.rho: $(newMarginal.rho), msgBackX.rho: $(msgBackX.rho))")
	end
	newMsg = newMarginal / msgBackX

	if isnan(new_approx_marginal_tau_rho[1]) || isnan(new_approx_marginal_tau_rho[2]) || !isfinite(msgBackX.rho) || !isfinite(msgBackX.tau) || !all(isfinite, new_approx_marginal_tau_rho)
		error("Number is nan: ", new_approx_marginal_tau_rho, " given this input: ", input, " and newMsg: ", newMsg, " and msgBackX: ", msgBackX)
	end



	# # debug code begin
	# analytical_new_message = update_msg_to_x(to_mean_variance(input)...)
	# if abs(GaussianDistribution.mean(analytical_new_message) - GaussianDistribution.mean(newMsg)) > 0.1*abs(GaussianDistribution.mean(analytical_new_message))
	#   println("====================================")
	#   println("mean discrepancy in msgto x: ", abs(GaussianDistribution.mean(analytical_new_message) - GaussianDistribution.mean(newMsg)))
	#   println("analytical message mean, var, tau, rho: ", GaussianDistribution.mean(analytical_new_message), " ", GaussianDistribution.variance(analytical_new_message), " ", analytical_new_message.tau, " ", analytical_new_message.rho)
	#   println("clamped approximate message mean, var, tau, rho: ", GaussianDistribution.mean(newMsg), " ", GaussianDistribution.variance(newMsg), " ", newMsg.tau, " ", newMsg.rho)
	#   unclamped_approx_msg = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2]) / msgBackX
	#   println("unclamped approximate message mean, var, tau, rho: ", GaussianDistribution.mean(unclamped_approx_msg), " ", GaussianDistribution.variance(unclamped_approx_msg), " ", unclamped_approx_msg.tau, " ", unclamped_approx_msg.rho)
	#   println("input tau, rho: ", input)
	#   println("input mean, var: ", to_mean_variance(input))
	#   println("====================================")
	# end
	# # debug code end

	oldMarginal = f.db[f.x]
	f.db[f.x] = newMarginal
	f.db[f.msg_to_x] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

# function update_msg_to_y!(f::ApproximateWeightedSumFactor)
#   msgBackY = f.db[f.y] / f.db[f.msg_to_y]
#   msgBackZ = f.db[f.z] / f.db[f.msg_to_z]
#   msgBackX = f.db[f.x] / f.db[f.msg_to_x]

#   input = [msgBackX.tau, msgBackX.rho, msgBackY.tau, msgBackY.rho, msgBackZ.tau, msgBackZ.rho, f.a, f.b, f.c]
#   new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_y, input)
#   newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2])

#   # Clamp variance of marginal to avoid negative rho in newMsg.
#   # Convert to mean/var first, since, if we only clamp rho, tau will be distorted (because it is a function of both mean and var).
#   # Use prevfloat to make sure the variance does not get slightly bigger due to numerical errors (i.e. inversion of floats not being an involution).
#   new_approx_marginal_mean = GaussianDistribution.mean(newMarginal)
#   new_approx_marginal_variance = clamp(GaussianDistribution.variance(newMarginal), 0.0, prevfloat(GaussianDistribution.variance(msgBackY)))
#   newMarginal = Gaussian1DFromMeanVariance(new_approx_marginal_mean, new_approx_marginal_variance)
#   # newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], clamp(new_approx_marginal_tau_rho[2], msgBackY.rho, Inf))

#   # # debug code begin
#   # analytical_new_marginal = update_marginal_y(to_mean_variance(input)...)
#   # if abs(GaussianDistribution.mean(analytical_new_marginal) - GaussianDistribution.mean(newMarginal)) > 0.1*abs(GaussianDistribution.mean(analytical_new_marginal))
#   #   println("====================================")
#   #   println("mean discrepancy in marginal y: ", abs(GaussianDistribution.mean(analytical_new_marginal) - GaussianDistribution.mean(newMarginal)))
#   #   println("analytical marginal mean, var, tau, rho: ", GaussianDistribution.mean(analytical_new_marginal), " ", GaussianDistribution.variance(analytical_new_marginal), " ", analytical_new_marginal.tau, " ", analytical_new_marginal.rho)
#   #   println("clamped approximate marginal mean, var, tau, rho: ", GaussianDistribution.mean(newMarginal), " ", GaussianDistribution.variance(newMarginal), " ", newMarginal.tau, " ", newMarginal.rho)
#   #   println("unclamped approximate marginal mean, var, tau, rho: ", new_approx_marginal_tau_rho[1] / new_approx_marginal_tau_rho[2], " ", 1 / new_approx_marginal_tau_rho[2], " ", new_approx_marginal_tau_rho[1], " ", new_approx_marginal_tau_rho[2])
#   #   println("input tau, rho: ", input)
#   #   println("input mean, var: ", to_mean_variance(input))
#   #   println("====================================")
#   # end
#   # # debug code end

#   newMsg = newMarginal / msgBackY

#   # # debug code begin
#   # analytical_new_message = update_msg_to_y(to_mean_variance(input)...)
#   # if abs(GaussianDistribution.mean(analytical_new_message) - GaussianDistribution.mean(newMsg)) > 0.1*abs(GaussianDistribution.mean(analytical_new_message))
#   #   println("====================================")
#   #   println("mean discrepancy in msgto y: ", abs(GaussianDistribution.mean(analytical_new_message) - GaussianDistribution.mean(newMsg)))
#   #   println("analytical message mean, var, tau, rho: ", GaussianDistribution.mean(analytical_new_message), " ", GaussianDistribution.variance(analytical_new_message), " ", analytical_new_message.tau, " ", analytical_new_message.rho)
#   #   println("clamped approximate message mean, var, tau, rho: ", GaussianDistribution.mean(newMsg), " ", GaussianDistribution.variance(newMsg), " ", newMsg.tau, " ", newMsg.rho)
#   #   unclamped_approx_msg = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2]) / msgBackY
#   #   println("unclamped approximate message mean, var, tau, rho: ", GaussianDistribution.mean(unclamped_approx_msg), " ", GaussianDistribution.variance(unclamped_approx_msg), " ", unclamped_approx_msg.tau, " ", unclamped_approx_msg.rho)
#   #   println("input tau, rho: ", input)
#   #   println("input mean, var: ", to_mean_variance(input))
#   #   println("====================================")
#   # end
#   # # debug code end

#   oldMarginal = f.db[f.y]
#   f.db[f.y] = newMarginal
#   f.db[f.msg_to_y] = newMsg
#   return (absdiff(oldMarginal, newMarginal))
# end

function update_msg_to_y!(f::ApproximateWeightedSumFactor)
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]


	input = [msgBackX.tau, msgBackX.rho, msgBackY.tau, msgBackY.rho, msgBackZ.tau, msgBackZ.rho, f.a, f.b, f.c]
	new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_y_1st, input)[1], predict_sample(nn_wsf_marginal_y_2nd, input)[1]
	#new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_y, input)
	newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2])

	"""
	newMsgY = Gaussian1DFromMeanVariance(
		GaussianDistribution.mean(msgBackZ) / f.b - f.a / f.b * GaussianDistribution.mean(msgBackX) - f.c  / f.b,
		GaussianDistribution.variance(msgBackZ) / (f.b * f.b) + f.a * f.a / (f.b * f.b) * GaussianDistribution.variance(msgBackX),
	)
	"""
	if newMarginal.rho < msgBackY.rho
		error("newMarginal.rho < msgBackY.rho (newMarginal.rho: $(newMarginal.rho), msgBackY.rho: $(msgBackY.rho))")
	end
	newMsg = newMarginal / msgBackY

	if isnan(new_approx_marginal_tau_rho[1]) || isnan(new_approx_marginal_tau_rho[2]) || !isfinite(msgBackY.rho) || !isfinite(msgBackY.tau) || !all(isfinite, new_approx_marginal_tau_rho)
		error("Number is nan: ", new_approx_marginal_tau_rho, " given this input: ", input, " and newMsg: ", newMsg, " and msgBackY: ", msgBackY)
	end


	oldMarginal = f.db[f.y]
	f.db[f.y] = newMarginal
	f.db[f.msg_to_y] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

# function update_msg_to_z!(f::ApproximateWeightedSumFactor)
#   msgBackY = f.db[f.y] / f.db[f.msg_to_y]
#   msgBackZ = f.db[f.z] / f.db[f.msg_to_z]
#   msgBackX = f.db[f.x] / f.db[f.msg_to_x]

#   input = [msgBackX.tau, msgBackX.rho, msgBackY.tau, msgBackY.rho, msgBackZ.tau, msgBackZ.rho, f.a, f.b, f.c]
#   new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_z, input)
#   newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2])

#   # Clamp variance of marginal to avoid negative rho in newMsg.
#   # Convert to mean/var first, since, if we only clamp rho, tau will be distorted (because it is a function of both mean and var).
#   # Use prevfloat to make sure the variance does not get slightly bigger due to numerical errors (i.e. inversion of floats not being an involution).
#   new_approx_marginal_mean = GaussianDistribution.mean(newMarginal)
#   new_approx_marginal_variance = clamp(GaussianDistribution.variance(newMarginal), 0.0, prevfloat(GaussianDistribution.variance(msgBackZ)))
#   newMarginal = Gaussian1DFromMeanVariance(new_approx_marginal_mean, new_approx_marginal_variance)
#   # newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], clamp(new_approx_marginal_tau_rho[2], msgBackZ.rho, Inf))

#   # # debug code begin
#   # analytical_new_marginal = update_marginal_z(to_mean_variance(input)...)
#   # if abs(GaussianDistribution.mean(analytical_new_marginal) - GaussianDistribution.mean(newMarginal)) > 0.1*abs(GaussianDistribution.mean(analytical_new_marginal))
#   #   println("====================================")
#   #   println("mean discrepancy in marginal z: ", abs(GaussianDistribution.mean(analytical_new_marginal) - GaussianDistribution.mean(newMarginal)))
#   #   println("analytical marginal mean, var, tau, rho: ", GaussianDistribution.mean(analytical_new_marginal), " ", GaussianDistribution.variance(analytical_new_marginal), " ", analytical_new_marginal.tau, " ", analytical_new_marginal.rho)
#   #   println("clamped approximate marginal mean, var, tau, rho: ", GaussianDistribution.mean(newMarginal), " ", GaussianDistribution.variance(newMarginal), " ", newMarginal.tau, " ", newMarginal.rho)
#   #   println("unclamped approximate marginal mean, var, tau, rho: ", new_approx_marginal_tau_rho[1] / new_approx_marginal_tau_rho[2], " ", 1 / new_approx_marginal_tau_rho[2], " ", new_approx_marginal_tau_rho[1], " ", new_approx_marginal_tau_rho[2])
#   #   println("input tau, rho: ", input)
#   #   println("input mean, var: ", to_mean_variance(input))
#   #   println("====================================")
#   # end
#   # # debug code end

#   newMsg = newMarginal / msgBackZ

#   # # debug code begin
#   # analytical_new_message = update_msg_to_z(to_mean_variance(input)...)
#   # if abs(GaussianDistribution.mean(analytical_new_message) - GaussianDistribution.mean(newMsg)) > 0.1*abs(GaussianDistribution.mean(analytical_new_message))
#   #   println("====================================")
#   #   println("mean discrepancy in msgto z: ", abs(GaussianDistribution.mean(analytical_new_message) - GaussianDistribution.mean(newMsg)))
#   #   println("analytical message mean, var, tau, rho: ", GaussianDistribution.mean(analytical_new_message), " ", GaussianDistribution.variance(analytical_new_message), " ", analytical_new_message.tau, " ", analytical_new_message.rho)
#   #   println("clamped approximate message mean, var, tau, rho: ", GaussianDistribution.mean(newMsg), " ", GaussianDistribution.variance(newMsg), " ", newMsg.tau, " ", newMsg.rho)
#   #   unclamped_approx_msg = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2]) / msgBackZ
#   #   println("unclamped approximate message mean, var, tau, rho: ", GaussianDistribution.mean(unclamped_approx_msg), " ", GaussianDistribution.variance(unclamped_approx_msg), " ", unclamped_approx_msg.tau, " ", unclamped_approx_msg.rho)
#   #   println("input tau, rho: ", input)
#   #   println("input mean, var: ", to_mean_variance(input))
#   #   println("====================================")
#   # end
#   # # debug code end

#   oldMarginal = f.db[f.z]
#   f.db[f.z] = newMarginal
#   f.db[f.msg_to_z] = newMsg
#   return (absdiff(oldMarginal, newMarginal))
# end

function update_msg_to_z!(f::ApproximateWeightedSumFactor)
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]

	"""
	input = [msgBackX.tau, msgBackX.rho, msgBackY.tau, msgBackY.rho, msgBackZ.tau, msgBackZ.rho, f.a, f.b, f.c]
	new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_z_1st, input)[1], predict_sample(nn_wsf_marginal_z_2nd, input)[1]
	#new_approx_marginal_tau_rho = predict_sample(nn_wsf_marginal_z, input)
	newMarginal = Gaussian1D(new_approx_marginal_tau_rho[1], new_approx_marginal_tau_rho[2])
	"""


	newMsgZ = Gaussian1DFromMeanVariance(
		f.a * GaussianDistribution.mean(msgBackX) + f.b * GaussianDistribution.mean(msgBackY) + f.c,
		f.a * f.a * GaussianDistribution.variance(msgBackX) + f.b * f.b * GaussianDistribution.variance(msgBackY),
	)

	"""
	if newMarginal.rho < msgBackZ.rho
	  error("newMarginal.rho < msgBackZ.rho (newMarginal.rho: (newMarginal.rho), msgBackZ.rho: $(msgBackZ.rho))")
	end
	newMsg = newMarginal / msgBackZ

	if isnan(new_approx_marginal_tau_rho[1]) || isnan(new_approx_marginal_tau_rho[2]) || !isfinite(msgBackZ.rho) || !isfinite(msgBackZ.tau) || !all(isfinite, new_approx_marginal_tau_rho)
	  error("Number is nan: ", new_approx_marginal_tau_rho, " given this input: ", input, " and newMsg: ", newMsg, " and msgBackZ: ", msgBackZ)
	end
	"""

	oldMarginal = f.db[f.z]
	newMarginal = oldMarginal / f.db[f.msg_to_z] * newMsgZ
	f.db[f.z] = newMarginal
	f.db[f.msg_to_z] = newMsgZ
	return (absdiff(oldMarginal, newMarginal))
end

struct GreaterThanFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	eps::Float64
	msg_to_x::Int64
end

# Initializes the greater than factor. Eps is the value that should be used for comparison, it is set to 0.0 at all times for now.
GreaterThanFactor(db::DistributionBag{Gaussian1D}, x::Int64, eps = 0.0) = GreaterThanFactor(db, x, eps, add!(db))

d = Distributions.Normal()

# computes the additive correction of a single-sided truncated Gaussian with unit variance
function v(t, ϵ)
	denom = Distributions.cdf(d, t - ϵ)
	if (denom < floatmin(Float64))
		return (ϵ - t)
	else
		return (Distributions.pdf(d, t - ϵ) / denom)
	end
end

# computes the multiplicative correction of a single-sided truncated Gaussian with unit variance
function w(t, ϵ)
	denom = Distributions.cdf(d, t - ϵ)
	if (denom < floatmin(Float64))
		return ((t - ϵ < 0.0) ? 1.0 : 0.0)
	else
		vt = v(t, ϵ)
		return (vt * (vt + t - ϵ))
	end
end

function update_msg_to_x!(f::GreaterThanFactor)
	msgBack = f.db[f.x] / f.db[f.msg_to_x]

	a = msgBack.tau / sqrt(msgBack.rho)
	b = f.eps * sqrt(msgBack.rho)
	c = 1.0 - w(a, b)
	newMarginal = Gaussian1D(
		(msgBack.tau + sqrt(msgBack.rho) * v(a, b)) / c,
		msgBack.rho / c,
	)
	oldMarginal = f.db[f.x]
	f.db[f.msg_to_x] = newMarginal / msgBack
	f.db[f.x] = newMarginal
	return (absdiff(oldMarginal, newMarginal))
end

function normalize_log_var!(f::GreaterThanFactor)
	logZ = logNormProduct(f.db[f.x], f.db[f.msg_to_x])
	f.db[f.x] *= f.db[f.msg_to_x]
	return (logZ)
end

function normalize_log_factor!(f::GreaterThanFactor)
	msgBack = f.db[f.x] / f.db[f.msg_to_x]
	logZ = -logNormProduct(msgBack, f.db[f.msg_to_x]) +
		   log(Distributions.cdf(d, (GaussianDistribution.mean(msgBack) - f.eps) / sqrt(GaussianDistribution.variance(msgBack))))
	return (logZ)
end

function normalize!(factorList::Vector{Factor}, db::DistributionBag, variableList::Vector{Int})
	# reset all the variables to the prior
	# reset!(db)
	# for i in variableList
	#   db.bag[i] = db.uniform
	# end
	reset!(db, variableList)

	logZ = 0
	# re-compute the marginals of all factors
	for i in eachindex(factorList)
		logZ += normalize_log_var!(factorList[i])
	end

	# re-normalize the factors 
	for i in eachindex(factorList)
		logZ += normalize_log_factor!(factorList[i])
	end

	return (logZ)
end

struct ApproximateEMFactor <: Factor
	db::DistributionBag{Gaussian1D}
	x::Int64
	y::Int64
	z::Int64
	q0::Float64
	dt::Float64
	msg_to_x::Int64
	msg_to_y::Int64
	msg_to_z::Int64
	nn_x::Any
	nn_y::Any
	nn_z::Any
	multiple::Bool
	sampling::Bool
end


ApproximateEMFactor(db::DistributionBag{Gaussian1D}, x::Int64, y::Int64, z::Int64, q0::Float64, dt::Float64, nn_x, nn_y, nn_z, multiple, sampling) =
	ApproximateEMFactor(db, x, y, z, q0, dt, add!(db), add!(db), add!(db), nn_x, nn_y, nn_z, multiple, sampling)


function update_msg_to_x!(f::ApproximateEMFactor)
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]

	# Define a maximum variance threshold to replace Inf
	max_variance = f.sampling ? Inf : 1e10

	# Ensure variances are finite
	varX = isinf(variance(msgBackX)) ? max_variance : variance(msgBackX)
	varY = isinf(variance(msgBackY)) ? max_variance : variance(msgBackY)
	varZ = isinf(variance(msgBackZ)) ? max_variance : variance(msgBackZ)

	input = [
		GaussianDistribution.mean(msgBackX),
		varX,
		GaussianDistribution.mean(msgBackY),
		varY,
		GaussianDistribution.mean(msgBackZ),
		varZ,
		f.q0,  # Assuming f.q0 is already finite
		f.dt,   # Assuming f.dt is already finite
	]

	if f.sampling
		# write input to file
		open("input_X.csv", "a") do io
			write(io, join(input, ",") * "\n")
		end
	end

	if f.multiple

		predicted_means = []
		predicted_vars = []

		for model in f.nn_x
			predicted_mean, predicted_var = predict_sample(model, input)
			push!(predicted_means, predicted_mean)
			push!(predicted_vars, predicted_var)
		end

		mean = StatsBase.mean(predicted_means)
		var = StatsBase.mean(predicted_vars)
		newMarginal = [mean, var]
	else
		if f.sampling
			while true
				try
					newMarginal, _, _ = generate_output_em_factor(input, variance_relative_epsilon = 1e-2)
					newMarginal = newMarginal[1:2]
					break
				catch e
					println("Error: ", e)
				end
			end
		else
			mean, var = predict_sample(f.nn_x, input)
			newMarginal = [mean, var]
		end
	end

	if f.sampling
		open("output_X.csv", "a") do io
			write(io, join(newMarginal, ",") * "\n")
		end
	end

	newMarginal = Gaussian1DFromMeanVariance(newMarginal...)

	if newMarginal.rho < msgBackX.rho
		error("newMarginal.rho < msgBackX.rho (newMarginal: $(newMarginal), msgBackX: $(msgBackX), input: $(input))")
	end
	newMsg = newMarginal / msgBackX

	oldMarginal = f.db[f.x]
	f.db[f.x] = newMarginal
	f.db[f.msg_to_x] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

function update_msg_to_y!(f::ApproximateEMFactor)
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]


	# Define a maximum variance threshold to replace Inf
  	max_variance = f.sampling ? Inf : 1e10

	# Ensure variances are finite
	varX = isinf(variance(msgBackX)) ? max_variance : variance(msgBackX)
	varY = isinf(variance(msgBackY)) ? max_variance : variance(msgBackY)
	varZ = isinf(variance(msgBackZ)) ? max_variance : variance(msgBackZ)

	input = [
		GaussianDistribution.mean(msgBackX),
		varX,
		GaussianDistribution.mean(msgBackY),
		varY,
		GaussianDistribution.mean(msgBackZ),
		varZ,
		f.q0,  # Assuming f.q0 is already finite
		f.dt,   # Assuming f.dt is already finite
	]

	# write input to file
	# write input to file as CSV
	if f.sampling
		open("input_Y.csv", "a") do io
			write(io, join(input, ",") * "\n")
		end
	end

	#println("Update msg to y")

	if f.multiple

		predicted_means = []
		predicted_vars = []

		for model in f.nn_y
			predicted_mean, predicted_var = predict_sample(model, input)
			push!(predicted_means, predicted_mean)
			push!(predicted_vars, predicted_var)
		end

		mean = StatsBase.mean(predicted_means)
		var = StatsBase.mean(predicted_vars)
		newMarginal = [mean, var]
	else

		if f.sampling
			while true
				try
					newMarginal, _, _ = generate_output_em_factor(input, variance_relative_epsilon = 1e-2)
					newMarginal = newMarginal[3:4]
					break
				catch e
					println("Error: ", e)
				end
			end
		else
			mean, var = predict_sample(f.nn_y, input)
			newMarginal = [mean, var]
		end
	end

	if f.sampling
		open("output_Y.csv", "a") do io
			write(io, join(newMarginal, ",") * "\n")
		end
	end

	newMarginal = Gaussian1DFromMeanVariance(newMarginal[1], newMarginal[2])
	if newMarginal.rho < msgBackY.rho
		error("newMarginal.rho < msgBackY.rho (newMarginal.rho: $(newMarginal.rho), msgBackY.rho: $(msgBackY.rho))")
	end
	newMsg = newMarginal / msgBackY

	oldMarginal = f.db[f.y]
	f.db[f.y] = newMarginal
	f.db[f.msg_to_y] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

function update_msg_to_z!(f::ApproximateEMFactor)
	msgBackX = f.db[f.x] / f.db[f.msg_to_x]
	msgBackY = f.db[f.y] / f.db[f.msg_to_y]
	msgBackZ = f.db[f.z] / f.db[f.msg_to_z]

	# if sampling, then leave Inf, if nets, then set high variance
	max_variance = f.sampling ? Inf : 1e10

	# Ensure variances are finite
	varX = isinf(variance(msgBackX)) ? max_variance : variance(msgBackX)
	varY = isinf(variance(msgBackY)) ? max_variance : variance(msgBackY)
	varZ = isinf(variance(msgBackZ)) ? max_variance : variance(msgBackZ)

	input = [
		GaussianDistribution.mean(msgBackX),
		varX,
		GaussianDistribution.mean(msgBackY),
		varY,
		GaussianDistribution.mean(msgBackZ),
		varZ,
		f.q0,  # Assuming f.q0 is already finite
		f.dt,   # Assuming f.dt is already finite
	]

	# write input to file as CSV
	if f.sampling
		open("input_Z.csv", "a") do io
			write(io, join(input, ",") * "\n")
		end
	end


	if f.multiple

		predicted_means = []
		predicted_vars = []

		for model in f.nn_z
			predicted_mean, predicted_var = predict_sample(model, input)
			push!(predicted_means, predicted_mean)
			push!(predicted_vars, predicted_var)
		end

		mean = StatsBase.mean(predicted_means)
		var = StatsBase.mean(predicted_vars)

		newMarginal = [mean, var]
	else
		#
		if f.sampling
			while true
				try
					newMarginal, _, _ = generate_output_em_factor(input, variance_relative_epsilon = 1e-2)
					newMarginal = newMarginal[5:6]
					break
				catch e
					println("Error: ", e)
				end
			end

		else

			mean, var = predict_sample(f.nn_z, input)
			newMarginal = [mean, var]
		end

	end


	if f.sampling
		open("output_Z.csv", "a") do io
			write(io, join(newMarginal, ",") * "\n")
		end
	end

	newMarginal = Gaussian1DFromMeanVariance(newMarginal[1], newMarginal[2])

	if newMarginal.rho < msgBackZ.rho
		error("newMarginal.rho < msgBackZ.rho (newMarginal.rho: (newMarginal.rho), msgBackZ.rho: $(msgBackZ.rho))")
	end
	newMsg = newMarginal / msgBackZ

	oldMarginal = f.db[f.z]
	f.db[f.z] = newMarginal
	f.db[f.msg_to_z] = newMsg
	return (absdiff(oldMarginal, newMarginal))
end

end
