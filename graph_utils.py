import networkx as nx


def make_graph(text):
    chunks = text.split()
    G = nx.DiGraph()
    for chunk in set(chunks):
        G.add_node(chunk)
    for i in range(len(chunks) - 1):
        G.add_edge(chunks[i], chunks[i + 1])
    return G


def graph_distance(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())
