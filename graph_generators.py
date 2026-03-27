import random
import networkx as nx
from networkx.algorithms import bipartite
import numpy as np
import subprocess
from deterministic_graph_algorithms import HavelHakimiAlgorithm

GRAPH_SIZES = np.arange(3, 12)

def generate_graphs(
        number_of_graphs: int,
        algorithm: str,
        weighted: bool,
        directed: bool,
        #random_seed: int = 1234,
        er_min_sparsity: float = 0.0,
        er_max_sparsity: float = 1.0,
        number_of_nodes=random.choice(GRAPH_SIZES),
) -> list[nx.Graph]:
    
    """Generates a list of graphs based on the algorithm provided and the 
        graph requirements given like weighted or directed"""

    #random.seed(random_seed)
    #np.random.seed(random_seed)

    generated_graphs = []
    #random_state = np.random.RandomState(random_seed)

    if algorithm == "er":
        while len(generated_graphs) < number_of_graphs:
            sparsity = random.uniform(er_min_sparsity, er_max_sparsity)
            G = nx.erdos_renyi_graph(number_of_nodes, sparsity, directed=directed)
            if weighted:
                G = add_weights(G)
            if directed:
                if nx.is_weakly_connected(G):
                    generated_graphs.append(G)
            else:
                if nx.is_connected(G):
                    generated_graphs.append(G)

    elif algorithm == "complete":
        while len(generated_graphs) < number_of_graphs:
            create_using = nx.DiGraph if directed else nx.Graph
            G = nx.complete_graph(number_of_nodes, create_using=create_using)
            if weighted:
                G = add_weights(G)
            if directed:
                if nx.is_weakly_connected(G):
                    generated_graphs.append(G)
            else:
                if nx.is_connected(G):
                    generated_graphs.append(G)

    elif algorithm == 'bipartite':
        while len(generated_graphs) < number_of_graphs:
            top_node = number_of_nodes
            bottom_node = random.choice(np.arange(3, 5))
            num_edges = random.randint(1, (top_node * bottom_node))
            B = bipartite.gnmk_random_graph(top_node, bottom_node, num_edges, seed=42, directed=directed)
            if weighted:
                B = add_weights(B)
            if directed:
                if nx.is_weakly_connected(B):
                    generated_graphs.append(B)
            else:
                if nx.is_connected(B):
                    generated_graphs.append(B)

    return generated_graphs


def add_weights(G):
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(1, 10)
    return G


def generate_sequences(number_of_sequences: int,
                       algorithm: str,
                       directed: bool,
                       weighted=False,
                       number_of_nodes=random.choice(GRAPH_SIZES)):
    """Generates a list graphical sequences by simpling returning 
        the degree of every node in the generated graph"""
    
    generated_sequences = []
    while number_of_sequences != 0:

        graph = generate_graphs(1, algorithm, weighted, directed, number_of_nodes)[0]
        generated_sequences.append(HavelHakimiAlgorithm.give_degree_sequence(graph))

        number_of_sequences -= 1

    return generated_sequences

def generate_graphs_n_nodes(
        number_of_nodes:int,
        weighted:bool,
        directed:bool,
        is_bipartite:bool,
        input_type:str ):   
    
    """Generates and returns a list of all non-isomorpic graphs of a particular number of nodes and 
        given number of edges starting from n-1 to complex graph, picking only one graph for a particular
        number of edges.
        For this purpose we will be using the geng command """

    command=['geng ']
    if  is_bipartite:
        command.append("-cb ")
    else:
        command.append("-c ")
    command.append(str(number_of_nodes))

    min_possible_edges=number_of_nodes-1
    max_possible_edges=(number_of_nodes*(number_of_nodes-1))/2
    
    for edge_count in range(min_possible_edges,max_possible_edges+1):
        graph_List=[]

        command.append(f"{edge_count}:{edge_count}")
        command.append("| shuf -n 1")
        
        results = subprocess.run(command, capture_output=True, text=True)
        results=results.stdout.strip().split("\n")

        graph=graph = nx.from_graph6_bytes(results[0].encode())

        if directed:
            graph = graph.to_directed()
        if weighted:
            graph = add_weights(graph)

        if input_type == 'graph': graph_List.append(graph)
        else : graph_List.append(HavelHakimiAlgorithm.give_degree_sequence(graph))
       
    return graph_List

    

