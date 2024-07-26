import networkx as nx
G = nx.DiGraph()
G.add_node(1)

import matplotlib.pyplot as plt


G = nx.DiGraph(
    [
        ("f", "a"),
        ("a", "b"),
        ("a", "e"),
        ("b", "c"),
        ("b", "d"),
        ("d", "e"),
        ("f", "c"),
        ("f", "g"),
        ("h", "f"),
    ]
)
G.add_node('f',d = 'damn')

print('G.predecessors(\'h\')')
#print(list(G.predecessors('h'))[0])
print(len(list(G.predecessors('h'))))




for layer, nodes in enumerate(nx.topological_generations(G)):
    # `multipartite_layout` expects the layer as a node attribute, so add the
    # numeric layer value as a node attribute
    for node in nodes:
        G.nodes[node]["layer"] = layer

# Compute the multipartite_layout using the "layer" node attribute
pos = nx.multipartite_layout(G, subset_key="layer")

fig, ax = plt.subplots()
nx.draw_networkx(G, ax=ax,pos=pos)
ax.set_title("DAG layout in topological order")
fig.tight_layout()
plt.show()