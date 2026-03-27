# for this particular paper we will be using adjacency encoding as described in 'Talk like a Graph Paper"
import networkx as nx


def create_node_string(name_dict, nnodes: int) -> str:
    node_string = ""
    sorted_keys = list(sorted(name_dict.keys()))
    for i in sorted_keys[: nnodes - 1]:
        node_string += name_dict[i] + ", "
    node_string += "and " + name_dict[sorted_keys[nnodes - 1]]
    return node_string


def unweighted_graph_encoder(graph: nx.Graph) -> str:
    name_dict = {x: str(x) for x in graph.nodes()}
    if graph.is_directed():
        output = (
            "In a directed graph, (u,v) means that there is an directed edge from node u"
            " to node v . "
        )
    else:
        output = (
            "In an undirected graph, (u,v) means that node u and node v are"
            " connected with an undirected edge. "
        )

    nodes_string = create_node_string(name_dict, len(graph.nodes()))
    output += "G describes a graph among nodes %s.\n" % nodes_string
    if graph.edges():
        output += "The edges in G are: "
    for i, j in graph.edges():
        output += "(%s, %s) " % (name_dict[i], name_dict[j])
    return output.strip() + ".\n"


def weighted_graph_encoder(graph: nx.Graph) -> str:
    name_dict = {x: str(x) for x in graph.nodes()}
    if graph.is_directed():
        output = (
            "In a directed weighted graph, (u,v,w) means that there is an edge from node u"
            " to node v of weight w. "
        )
    else:
        output = (
            "In an undirected graph, (u,v,w) means that node u and node v are"
            " connected with an undirected edge of weight w. "
        )

    nodes_string = create_node_string(name_dict, len(graph.nodes()))
    output += "G describes a graph among nodes %s.\n" % nodes_string
    if graph.edges():
        output += "The edges in G are: "
    for i, j, w in graph.edges(data=True):
        output += "(%s, %s, %d) " % (name_dict[i], name_dict[j], w['weight'])
    return output.strip() + ".\n"

