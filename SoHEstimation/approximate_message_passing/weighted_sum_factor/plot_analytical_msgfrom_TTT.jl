using JLD2
using Plots

function max_every(vec::AbstractVector, interval::Int=1000)
    result = []
    for i in 1:1000:length(vec)
        end_index = min(i + 999, length(vec))
        push!(result, maximum(vec[i:end_index]))
    end
    return result
end

# log = load("/Users/leonhard.hennicke/Documents/phd/phy-ml/SoHEstimation/approximate_message_passing/weighted_sum_factor/data/log_msgfrom_WSF_TTT_2_68-560-686.jld2")

# msgfromx_rhos = log["msgfromx_rhos"]
# msgfromx_taus = log["msgfromx_taus"]
# msgfromy_rhos = log["msgfromy_rhos"]
# msgfromy_taus = log["msgfromy_taus"]
# msgfromz_rhos = log["msgfromz_rhos"]
# msgfromz_taus = log["msgfromz_taus"]

# msgfromx_vars = 1 ./ msgfromx_rhos
# msgfromy_vars = 1 ./ msgfromy_rhos
# msgfromz_vars = 1 ./ msgfromz_rhos

# msgfromx_stds = sqrt.(msgfromx_vars)
# msgfromy_stds = sqrt.(msgfromy_vars)
# msgfromz_stds = sqrt.(msgfromz_vars)

# msgfromx_means = msgfromx_taus ./ msgfromx_rhos
# msgfromy_means = msgfromy_taus ./ msgfromy_rhos
# msgfromz_means = msgfromz_taus ./ msgfromz_rhos

# jldsave(
#     "/Users/leonhard.hennicke/Documents/phd/phy-ml/SoHEstimation/approximate_message_passing/weighted_sum_factor/data/extended_log_msgfrom_WSF_TTT_2_68-560-686.jld2",
#     msgfromx_rhos = msgfromx_rhos,
#     msgfromy_rhos = msgfromy_rhos,
#     msgfromz_rhos = msgfromz_rhos,

#     msgfromx_taus = msgfromx_taus,
#     msgfromy_taus = msgfromy_taus,
#     msgfromz_taus = msgfromz_taus,

#     msgfromx_vars = msgfromx_vars,
#     msgfromy_vars = msgfromy_vars,
#     msgfromz_vars = msgfromz_vars,

#     msgfromx_stds = msgfromx_stds,
#     msgfromy_stds = msgfromy_stds,
#     msgfromz_stds = msgfromz_stds,

#     msgfromx_means = msgfromx_means,
#     msgfromy_means = msgfromy_means,
#     msgfromz_means = msgfromz_means,
# )


log = load("/Users/leonhard.hennicke/Documents/phd/phy-ml/SoHEstimation/approximate_message_passing/weighted_sum_factor/data/log_msgfrom_WSF_TTT_3.jld2")

msgfromx_rhos = log["msgfromx_rhos"]
msgfromy_rhos = log["msgfromy_rhos"]
msgfromz_rhos = log["msgfromz_rhos"]

msgfromx_taus = log["msgfromx_taus"]
msgfromy_taus = log["msgfromy_taus"]
msgfromz_taus = log["msgfromz_taus"]

msgfromx_vars = log["msgfromx_vars"]
msgfromy_vars = log["msgfromy_vars"]
msgfromz_vars = log["msgfromz_vars"]

msgfromx_stds = log["msgfromx_stds"]
msgfromy_stds = log["msgfromy_stds"]
msgfromz_stds = log["msgfromz_stds"]

msgfromx_means = log["msgfromx_means"]
msgfromy_means = log["msgfromy_means"]
msgfromz_means = log["msgfromz_means"]

update_log = log["update_log"]

println("loaded log")

every = 100_000

# # plot the histograms rhos
# p1 = []

# for (rho, v) in zip([msgfromx_rhos, msgfromy_rhos, msgfromz_rhos], ["x", "y", "z"])
#     push!(p1, histogram(rho, title="msgfrom $v rho"))
# end
# push!(p1, histogram(vcat(msgfromx_rhos, msgfromy_rhos, msgfromz_rhos), title="msgfrom xyz rho"))

# p = plot(p1...)
# display(p)


# # plot the histograms taus
# p1 = []

# for (tau, v) in zip([msgfromx_taus, msgfromy_taus, msgfromz_taus], ["x", "y", "z"])
#     push!(p1, histogram(tau, title="msgfrom $v tau hist"))
# end
# push!(p1, histogram(vcat(msgfromx_taus, msgfromy_taus, msgfromz_taus), title="msgfrom xyz tau hist"))

# p = plot(p1...)
# display(p)

############################

# # plot the lineplots of the stds
# p1 = []

# for (std, v) in zip([msgfromx_stds, msgfromy_stds, msgfromz_stds], ["x", "y", "z"])
#     push!(p1, plot(max_every(std, every), title="msgfrom $v std progression (max every $every)"))
# end

# p = plot(p1..., layout=(3,1))
# display(p)

# # plot the lineplots of the means
# p1 = []

# for (m, v) in zip([msgfromx_means, msgfromy_means, msgfromz_means], ["x", "y", "z"])
#     push!(p1, plot(max_every(m, every), title="msgfrom $v mean progression (max every $every)"))
# end

# p = plot(p1..., layout=(3,1))
# display(p)

############################

# # plot histograms stds

# p1 = []

# for (std, v) in zip([msgfromx_stds, msgfromy_stds, msgfromz_stds], ["x", "y", "z"])
#     push!(p1, histogram(std, title="msgfrom $v std histogram"))
# end
# push!(p1, histogram(vcat(msgfromx_stds, msgfromy_stds, msgfromz_stds), title="msgfrom xyz std histogram"))

# p = plot(p1..., layout=(4,1))
# display(p)


# # plot the histograms means
# p1 = []

# for (tau, v) in zip([msgfromx_means, msgfromy_means, msgfromz_means], ["x", "y", "z"])
#     push!(p1, histogram(tau, title="msgfrom $v mean histogram"))
# end
# push!(p1, histogram(vcat(msgfromx_means, msgfromy_means, msgfromz_means), title="msgfrom xyz mean histogram"))

# p = plot(p1..., layout=(4,1))
# display(p)


# # plot histograms stds < 1

# p1 = []

# for (std, v) in zip([filter(<(1), msgfromx_stds), filter(<(1), msgfromy_stds), filter(<(1), msgfromz_stds)], ["x", "y", "z"])
#     push!(p1, histogram(std, title="msgfrom $v std < 1 histogram"))
# end
# push!(p1, histogram(vcat(filter(<(1), msgfromx_stds), filter(<(1), msgfromy_stds), filter(<(1), msgfromz_stds)), title="msgfrom xyz std < 1 histogram"))

# p = plot(p1..., layout=(4,1))
# display(p)


# # plot the histograms means < 1
# p1 = []

# for (tau, v) in zip([filter(<(1), msgfromx_means), filter(<(1), msgfromy_means), filter(<(1), msgfromz_means)], ["x", "y", "z"])
#     push!(p1, histogram(tau, title="msgfrom $v mean < 1 histogram"))
# end
# push!(p1, histogram(vcat(filter(<(1), msgfromx_means), filter(<(1), msgfromy_means), filter(<(1), msgfromz_means)), title="msgfrom xyz mean < 1 histogram"))

# p = plot(p1..., layout=(4,1))
# display(p)