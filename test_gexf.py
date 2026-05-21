import networkx as nx

g = nx.DiGraph()
g.add_node(1, viz={'color': {'r': 255, 'g': 0, 'b': 0, 'a': 1.0}, 'position': {'x': 10.0, 'y': 20.0, 'z': 0.0}, 'size': 50.0})
g.add_node(2, viz={'color': {'r': 0, 'g': 255, 'b': 0, 'a': 1.0}, 'position': {'x': 50.0, 'y': 20.0, 'z': 0.0}, 'size': 25.0})
g.add_edge(1, 2, weight=2.0)
nx.write_gexf(g, "test.gexf")
