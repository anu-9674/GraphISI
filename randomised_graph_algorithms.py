"""
This file contains the implementations of the randomised graph algorithms that will be used for the study
"""
import random
import json
import networkx as nx
from deterministic_graph_algorithms import GraphAlgorithms
from GraphAlgorithms import DisjointSet


class KargerAlgorithm(GraphAlgorithms):
    """Implements Karger's Algorithm which is a randomised algorithm to find the min cut of a graph"""

    def __init__(self, G):
        super().__init__(G)
        self.name = "Kargers_Algorithm"
        self.algorithm_steps = """
        """
        self.task = "Find the min cut of a graph using Karger's Algorithm"
        self.schema = """
        """
        self.Graph = G
        self.min_cut = []
        self.min_cut_size = 0
        self.edges = list(G.edges)
        self.num_vertices = len(G.nodes)

    def karger_algorithm(self):

        subsets = DisjointSet(self.Graph.nodes)
        num_vertices = self.num_vertices

        self.min_cut = []

        iteration = 1

        while num_vertices > 2:

            iteration_dict = {'current number of vertices': num_vertices}

            # we need to choose a random edge
            random_edge = random.choice(self.edges)

            # two corners of the current edge and the subset they belong to
            u, subset_u = random_edge[0], subsets.find_U_parent(random_edge[0])
            v, subset_v = random_edge[1], subsets.find_U_parent(random_edge[1])

            iteration_dict = {'edge_selected': {'u': u, 'v': v, 'subset u': subset_u, 'subset v': subset_v},
                              'do vertices of chosen edge belong to same subset': True,
                              'subset to which u and v are contracted to': None}

            if subset_v == subset_u:
                # the two vertices belong to the same component, so we need to consider a different edge
                continue

            else:
                # contract the edge
                iteration_dict['do vertices of chosen edge belong to same subset'] = False
                subsets.union_by_rank(u, v)
                iteration_dict['subset to which u and v are contracted to'] = subsets.find_U_parent(u)
                num_vertices -= 1

            self.algorithm_log[f"Iteration {iteration}"] = iteration_dict
            iteration += 1

        for edge in self.edges:
            u, subset_u = edge[0], subsets.find_U_parent(edge[0])
            v, subset_v = edge[1], subsets.find_U_parent(edge[1])
            if subset_u != subset_v:
                self.min_cut.append((u, v))

    
    def run(self):
        self.karger_algorithm()
        return [self.min_cut,len(self.min_cut)],self.algorithm_log


class RandomisedMST(GraphAlgorithms):
    """Randomised Minimum Spanning Tree Algorithm using Boruvka's Phase"""

    def __init__(self, G):
        super().__init__(G)

        self.name = 'Randomised Minimum Spanning Tree'
        self.algorithm_steps = """
        Algorithm MST :
        Input: Weighted, undirected graph G with n vertices and m edges.
        Output: Minimum spanning forest F for G .
        1 . Using three applications of Boruvka phases interleaved with simplification of
        the contracted graphs, compute a graph G1 with at most nj8 vertices and let
        C be the set of edges contracted during the three phases. If G is empty then
        exit and return F = C.
        2. Let G2 = G1 (p) be a randomly sampled subgraph of G1 , where p = 1/2.
        3. Recursively applying Algorithm MST, compute the minimum spanning forest
        F2 of the graph G2.
        4. Using a linear-time verification algorithm, identify the F2-heavy edges in G1
        and delete them to obtain a graph G3•
        5. Recursively applying Algorithm MST, compute the minimum spanning forest
        F3 for the graph G3•
        6. return forest F = C u F3•
        """
        self.task = "Find the minimum spanning tree of a graph using randomised algorithm that uses Boruvka's Phase"
        self.Graph = G
        self.contracted_graph = nx.Graph()
        self.subsets = DisjointSet
        self.mst = []
        self.mst_weight = 0
        self.max_edge_on_path = None

    def boruvka_phase(self):
        self.subsets = DisjointSet(self.Graph.nodes)
        numTrees = len(self.Graph.nodes)
        cheapest = [-1] * len(self.Graph.nodes)

        for edge in self.Graph.edges(data=True):
            u, v, w = edge[0], edge[1], edge[2]['weight']
            subset_u = self.subsets.find_U_parent(u)
            subset_v = self.subsets.find_U_parent(v)
            if subset_u != subset_v:
                if cheapest[subset_u] == -1 or cheapest[subset_u][2] > w:
                    cheapest[subset_u] = [u, v, w]
                if cheapest[subset_v] == -1 or cheapest[subset_v][2] > w:
                    cheapest[subset_v] = [u, v, w]

                # we will consider the above picked cheapest edges and add them to the MST
        for node in self.Graph.nodes:
            if cheapest[node] != -1:
                u, v, w = cheapest[node]
                subset_u = self.subsets.find_U_parent(u)
                subset_v = self.subsets.find_U_parent(v)

                if subset_v != subset_u:
                    self.mst_weight += w
                    self.mst.append([u, v, w])
                    self.subsets.union_by_rank(u, v)
        cheapest = [-1] * len(self.Graph.nodes)

    def contract_graph(self):
        component = {}
        for node in self.Graph.nodes:
            component[node] = self.subsets.find_U_parent(node=node)
        self.contracted_graph.add_nodes_from(set(component.values()))

        for u, v, data in self.Graph.edges(data=True):
            cu = component[u]
            cv = component[v]
            w = data['weight']
            if cu != cv:
                if self.contracted_graph.has_edge(cu, cv):
                    if self.contracted_graph[cu][cv]['weight'] > w:  # for nodes with multiple edges between them
                        self.contracted_graph[cu][cv]['weight'] = w
                else:
                    self.contracted_graph.add_edge(cu, cv, weight=w)
        return self.contracted_graph

    @staticmethod
    def sample_graph(G, p=0.5):
        Gs = nx.Graph()
        Gs.add_nodes_from(G.nodes)
        for u, v, data in G.edges(data=True):
            if random.random() <= p:
                Gs.add_edge(u, v, weight=data['weight'])
        return Gs

    @staticmethod
    def max_edge_on_path(T, u, v):
        path = nx.shortest_path(T, u, v)
        max_w = -float('inf')
        for i in range(len(path) - 1):
            w = T[path[i]][path[i + 1]]['weight']
            max_w = max(max_w, w)
        return max_w

    def remove_heavy_edges(self, G, F):
        T = nx.Graph()
        for u, v, w in F:
            T.add_edge(u, v, weight=w)

        G3 = nx.Graph()
        G3.add_nodes_from(G.nodes)

        for u, v, data in G.edges(data=True):
            if T.has_node(u) and T.has_node(v):
                max_w = self.max_edge_on_path(T, u, v)
                if data['weight'] <= max_w:
                    G3.add_edge(u, v, weight=data['weight'])
            else:
                G3.add_edge(u, v, weight=data['weight'])

        return G3

    def Karger_Klein_Tarjan_Algorithm(self):

        # Base case
        if self.Graph.number_of_edges() == 0:
            return []

        C = []

        # Step 1: Three Boruvka phases
        for _ in range(3):
            self.boruvka_phase()
            C.extend(self.mst)
            self.mst = []

        G1 = self.contract_graph()

        if G1.number_of_edges() == 0:
            return C

        # Step 2: Random sampling
        G2 = self.sample_graph(G1, p=0.5)

        # Step 3: Recursive MST on G2
        algo2 = RandomisedMST(G2)
        F2 = algo2.Karger_Klein_Tarjan_Algorithm()

        # Step 4: Remove heavy edges
        G3 = self.remove_heavy_edges(G1, F2)

        # Step 5: Recursive MST on G3
        algo3 = RandomisedMST(G3)
        F3 = algo3.Karger_Klein_Tarjan_Algorithm()

        # Step 6: Return union
        self.mst = C + F3
    
    def run(self):
        return self.mst, self.algorithm_log
