# plots.jl

using Plots
using StatsPlots

"""
    trace_plot(samples::Vector{ProjectileState}, variable::Symbol)

Create a trace plot for the specified variable from the given samples.
Valid variables are :x, :y, :velocity_x, and :velocity_y.
"""
function trace_plot(samples::Vector{ProjectileState}, variable::Symbol)
    values = getfield.(samples, variable)
    plot(values, title="Trace Plot for $variable", xlabel="Iteration", ylabel=string(variable))
end

"""
    trank_plot(samples::Vector{ProjectileState}, variable::Symbol)

Create a rank plot for the specified variable from the given samples.
Valid variables are :x, :y, :velocity_x, and :velocity_y.
"""
function trank_plot(samples::Vector{ProjectileState}, variable::Symbol)
    values = getfield.(samples, variable)
    ranks = sortperm(values)
    histogram(ranks, title="Rank Plot for $variable", xlabel="Rank", ylabel="Iteration", legend=false)
end

"""
    autocorrelation_plot(samples::Vector{ProjectileState}, variable::Symbol)

Create an autocorrelation plot for the specified variable from the given samples.
Valid variables are :x, :y, :velocity_x, and :velocity_y.

"""
function autocorrelation_plot(samples::Vector{ProjectileState}, variable::Symbol)
    values = getfield.(samples, variable)
    autocorrelation = StatsPlots.autocor(values)
    plot(autocorrelation, title="Autocorrelation Plot for $variable", xlabel="Lag", ylabel="Autocorrelation")
end
