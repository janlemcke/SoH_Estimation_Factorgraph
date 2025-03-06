# using Graphs
# using GraphPlot

# g = DiGraph()

# add_vertices!(g, 5)
# add_edge!(g,1,2)

# layout = spring_layout(g)

# gplot(g, layout, nodelabel=1:nv(g))

using Graphs
using GraphPlot

# Create a directed graph
g = DiGraph(5)  # Creates a graph with 5 nodes

# Add edges representing conditional dependencies
# Example: A -> B, A -> C, B -> D, C -> E
add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
add_edge!(g, 2, 4)
add_edge!(g, 3, 5)

# Layout for the graph plotting using spring layout
x, y = spring_layout(g)

# Plotting the graph with node labels
gplot(g, x, y, nodelabel=1:nv(g))
