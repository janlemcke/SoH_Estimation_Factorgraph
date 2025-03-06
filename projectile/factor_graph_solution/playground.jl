# include("../../lib/factors.jl")
# # include("../../lib/gaussian.jl")
# # include("../../lib/distributionbag.jl")


# using DataStructures
# using ..Factors
# using ..DistributionCollections
# using ..GaussianDistribution

# function printf(vars)
#     for var in vars
#         println("db[$var] = ", db[var], "\n")
#     end
#     println("----------------------")

# end

# db = DistributionBag(Gaussian1D(0, 0))

# x = add!(db)
# y = add!(db)
# z = add!(db)

# vars = [x,y,z]

# px = GaussianFactor(db, x, Gaussian1DFromMeanVariance(5.0, 4.0))
# py = GaussianFactor(db, y, Gaussian1DFromMeanVariance(5.0, 4.0))
# pz = GaussianFactor(db, z, Gaussian1DFromMeanVariance(9.0, 16.0))
# f = ScalingFactor(db, x,z,2.0, 1.0)
# f2 = WeightedSumFactor(db,x,y,z,2.0,2.0,1.0)

# update_msg_to_x!(px)
# update_msg_to_x!(py)
# # update_msg_to_x!(pz)
# printf(vars)
# update_msg_to_z!(f2)
# printf(vars)
# # update_msg_to_y!(f2)
# # printf(vars)
# # update_msg_to_x!(f2)
# # printf(vars)
# # update_msg_to_x!(f)
# # printf(vars)



# using DataStructures
# true_data = OrderedDict(0.0 => 1.0, 1.0 =>6.0, 2.0 => 11.0, 3.0 => 16.0, 4.0 => 21.0, 5.0 => 26.0, 6.0 => 31.0, 7.0 => 36.0)


# function y_v(t)
#     v_0 = 40.0
#     y_0 = 1.0

#     return y_0 + v_0*t + 0.5*(-9.8)*t^2, v_0 + t * (-9.8)
# end

# show(OrderedDict([(t, y_v(t)) for t in 0:7]))


using Plots, Random, Distributions, LinearAlgebra, DataStructures, ColorSchemes
function modify_alpha(cscheme::ColorScheme, alpha::Vector{T}, newname::String) where T<: AbstractFloat
    size(cscheme.colors, 1) == size(alpha, 1) ||
          error("Vector alpha must have the same size as colors")
    csalpha=ColorScheme([Colors.RGBA(c.r, c.g, c.b, a) for (c,a) in zip(cscheme.colors, alpha)], newname, "")
end
begin
ice_alpha=modify_alpha(ColorSchemes.ice, collect(range(1, 0, 256)), "mycscheme")
x = range(0, stop=2π, length=100)
y = range(0, stop=2π, length=100)
z = [sin(xi + yi) for xi in x, yi in y]
heatmap(x, y, z, color=cgrad(palette(ice_alpha, 256), rev=true), xlabel="X", ylabel="Y", title="Sine Heatmap", grid=true)
end