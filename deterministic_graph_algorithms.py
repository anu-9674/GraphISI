import heapq
import json
import random
import sys
import networkx as nx
import os
from collections import deque
from GraphAlgorithms import DisjointSet,GraphAlgorithms
from score import inversion_score


"""Implementation of various deterministic graph algorithms"""

class BFSAlgorithm(GraphAlgorithms):
    """Graph Algorithm for breadth first search"""

    def __init__(self, G):
        super().__init__()

        self.name = 'Breadth First Search'
        self.task = 'Find the breadth first search ordering of the nodes'
        self.algorithm_steps = """
                  BFS(G)
        1  for each vertex u in vertex set of Graph G do
        2      visited[u] = FALSE
        3  s = first vertex in vertex set of graph G
        4  visited[s] = TRUE
        5  Q = initialise empty queue
        6  ENQUEUE(Q, s)
        7 while Q is not empty do
        8     u = DEQUEUE(Q)
        9     add u to BFS ordering
        10     for each vertex v in adjacency list of vertex u do:
        11         if visited[v] = FALSE then
        12             visited[v] = TRUE
        13            ENQUEUE(Q, v)
        """
        self.schema = """
        {
            "BFS": list[integer],
            "steps": {
                "Iteration [0-9]+$": {
                "visited": list[integer],
                "state of queue": list[int],
                "current_node": integer,
                "current_node_adjacency_list": list[integer],
            }
        }
        """
        self.Graph = G
        self.bfs_node_ordering = []
        self.algorithm_log={}

    def bfs_algorithm(self):
        iteration = 1

        adjacency_list = super().give_adjacency_list(self.Graph)
        visited = [0] * len(self.Graph.nodes())

        source = list(self.Graph.nodes())[0]
        q = deque()

        visited[source] = 1
        q.append(source)

        self.algorithm_log[f'Iteration 0'] = {
            'visited': visited.copy(),
            'state_of_queue': list(q),
            'current_node': None,
            'current_node_neighborhood': None
        }

        while q:

            current_node = q.popleft()
            self.bfs_node_ordering.append(current_node)

            iteration_dict = {'visited': visited.copy(), 'state_of_queue': list(q), 'current_node': current_node,
                              'current_node_neighborhood': adjacency_list[current_node]}

            for node in adjacency_list[current_node]:
                if not visited[node]:
                    visited[node] = 1
                    q.append(node)

            self.algorithm_log[f'Iteration {iteration}'] = iteration_dict
            iteration += 1

    def run(self):
        self.bfs_algorithm()
        return self.bfs_node_ordering, self.algorithm_log
    
    def iteration_score(llm_iteration,gt_iteration):
        pass


class DFSAlgorithm(GraphAlgorithms):
    """Graph Algorithm for depth first search"""

    def __init__(self, G):
        super().__init__()

        self.name = 'Depth First Search'
        self.task = 'Find the depth first search ordering of the nodes'
        self.algorithm_steps = """
               DFS(G)
        1  for each vertex u that belongs to the vertex set of Graph G do:
        2      visited[u] = FALSE
        3  s = first vertex in vertex set of Graph G
        4  DFS-VISIT(s)
        
        DFS-VISIT(u)
        5  visited[u] = TRUE
        6  add u to DFS ordering
        7 for each vertex v that belongs to the adjacency list of vertex u do:
        8     if visited[v] = FALSE then
        9         DFS-VISIT(v)
        """
        self.schema = """
        {
            "DFS": list[integer],
            "steps": {
                "Iteration [0-9]+$": {
                "visited": list[integer],
                "state_of_stack": list[int],
                "current_node": integer,
                "current_node_neighborhood": list[integer],
            }
        }
           """
        self.algorithm_log = {}
        self.Graph = G
        self.dfs_node_ordering = []
        self.recursion_stack = []
        self.adjacency_list = super().give_adjacency_list(G)
        self.iteration = 1

    def dfs_recursive(self, visited, s):
        visited[s] = 1
        self.dfs_node_ordering.append(s)
        self.recursion_stack.append(s)

        self.algorithm_log[f'Iteration {self.iteration}'] = {
            'visited': visited.copy(),
            'state_of_stack': self.recursion_stack.copy(),
            'current_node': s,
            'current_node_neighborhood': self.adjacency_list[s]
        }
        self.iteration += 1

        for node in self.adjacency_list[s]:
            if not visited[node]:
                self.dfs_recursive(visited, node)

    def dfs_algorithm(self):
        visited = [0] * len(self.adjacency_list)
        self.dfs_recursive(visited, 0)

    def run(self):
        self.dfs_algorithm()
        return self.dfs_node_ordering, self.algorithm_log
    
    def iteration_score(llm_iteration,gt_iteration):
        pass


class DijkstraAlgorithm(GraphAlgorithms):
    """ Graph Algorithm for finding the shortest path from a single source"""

    def __init__(self, G):
        super().__init__()

        self.name = 'Dijkstra_Algorithm'
        self.task = "Find the shortest distances from a source using the following Algorithm"
        self.algorithm_steps = '''The steps of the Algorithm for a given graph is as follows: 
            DIJKSTRA(G, w, s)
        1   for each vertex v ∈ G.V
        2         d[v] = ∞
        3         π[v] = NIL
        4   d[s] = 0
        5   S = null
        6   Q = empty min-priority queue 
        7
        8   for each vertex v belong to the vertex set of graph G
        9        INSERT(Q, v, d[v])
        10  while Q is not empty
        11       u = EXTRACT-MIN(Q)
        12       Add u to S if u not in S
        13        for each vertex v in the adjacency list of vertex u:
        14              if d[v] > d[u] + w(u, v)
        15                   d[v] = d[u] + w(u, v)
        16                    π[v] = u
        17                    INSERT(Q,v,d[v])
        '''
        self.schema = """
        {
            "shortest path": list[integer],
            "steps": {
                "Iteration [0-9]+$": {
                "source": integer,
                "state of queue": list[list[integer,integer],
                "dist": {
                "u": integer,
                "v": integer,
                "weight": integer,
                "dist array":list[int],
            }
        }
        """
        self.Graph = G
        self.algorithm_log = {}

    def dijkstra_shortest_path_algorithm(self, adjacency_list):

        source = 0
        pq = []
        dist = [sys.maxsize] * len(self.Graph.nodes)

        heapq.heappush(pq, [0, source])  # weight,node
        dist[source] = 0

        iteration = 1

        while pq:

            iteration_dict = {"source": source, "state of queue": pq.copy(), "dist": []}

            u = heapq.heappop(pq)[1]

            for node in adjacency_list[u]:
                dist_dict = {}
                v, weight = node[0], node[1]

                if dist[v] > dist[u] + weight:
                    dist[v] = dist[u] + weight
                    heapq.heappush(pq, [dist[v], v])
                    dist_dict["u"] = u
                    dist_dict["v"] = v
                    dist_dict["weight"] = weight
                    dist_dict["dist array"] = dist.copy()
                    iteration_dict["dist"].append(dist_dict)

            self.algorithm_log[f"Iteration {iteration}"] = iteration_dict
            iteration += 1

        return dist

    def run(self):
        dist = self.dijkstra_shortest_path_algorithm(super().give_adjacency_list(self.Graph))
        return dist, self.algorithm_log

    @staticmethod
    def iteration_score(llm_iteration, gt_iteration):
        """
        This returns the score of every iteration by calculating the cumulative score of
        all the fields:
         source : once decided will remain the same across all iterations
                  This field will not be considered for evaluation
         state of queue : we have decided to add up the total number of inversions in the 2-element
                            arrays present in the queue
         dist : u : starting node (evaluation : 1 if same , 0 otherwise)
                v : node in the adjacency list (evaluation : 1 if same, 0 otherwise)
                w : weight associated with the [u,v] edge , (evaluation : 1 if same 0 if different)
                dist array : array of latest changes made to the distance array -
                             This field will be evaluated using our custom inversion based scoring metric
        """
        score_state_of_queue = inversion_score(llm_iteration['state of queue'], gt_iteration['state of queue'])

        if gt_iteration['dist'] == [] or llm_iteration['dist'] == []:
            return score_state_of_queue
        score_uvw, score_dist_array = 0, 0

        n = min(len(gt_iteration['dist']), len(llm_iteration['dist']))
        for i in range(n):
            score_uvw += ((1 if llm_iteration['dist'][i]['u'] == gt_iteration['dist'][i]['u'] else 0) + (
                1 if llm_iteration['dist'][i]['v'] == gt_iteration['dist'][i]['v'] else 0) + (
                              1 if llm_iteration['dist'][i]['weight'] == gt_iteration['dist'][i][
                                  'weight'] else 0)) / 3.0
            score_dist_array += inversion_score(llm_iteration['dist'][i]['dist array'],
                                                gt_iteration['dist'][i]['dist array'])

        score_uvw /= len(gt_iteration['dist'])
        score_dist_array /= len(gt_iteration['dist'])

        total_score = 0.6 * score_dist_array + 0.3 * score_state_of_queue + 0.1 * score_uvw

        return total_score
    
   


class HavelHakimiAlgorithm(GraphAlgorithms):
    """ Algorithm to find if a given degree sequence forms a graph or not"""

    def __init__(self, degree_sequence):
        super().__init__()

        self.name = "Havel_Hakimi"
        self.task = 'Finding if a given degree sequence forms a graph or not'
        self.algorithm_steps = '''
        The steps of the algorithm for a given degree sequence is as follows:
        Function IsGraphic(degreeSequence):
            While True:
                Sort degreeSequence in descending order  
                If all elements in degreeSequence is 0:
                    Return True
                Let d = degreeSequence[0]
                Remove degreeSequence[0]
                If d > Length of degreeSequence:
                    Return False
                For i from 0 to d-1:
                    degreeSequence[i] = degreeSequence[i] - 1
                    If degreeSequence[i] < 0:
                        Return False
        '''
        self.schema = """SCHEMA:
            {
            "result": integer,
            "steps": {
                "Iteration [0-9]+$": {
                "highest node degree:": integer,
                "reduced degree sequence": array[int],
                "is the sequence length smaller than highest degree ?":boolean
                "modified degree sequence": array[int],
                "is there a negative element":boolean,
            }"""
        self.degree_sequence = degree_sequence
        self.algorithm_log = {}

    @staticmethod
    def give_degree_sequence(Graph):
        degree_sequence = [Graph.degree(node) for node in Graph.nodes()]
        return degree_sequence

    def run(self):
        is_it_graph = self.havel_hakimi_algorithm()
        return is_it_graph, self.algorithm_log

    def havel_hakimi_algorithm(self):

        degree_sequence = self.degree_sequence.copy()

        # if the degree is more than the number of nodes then it is not a graphic sequence
        if any(num > len(degree_sequence) - 1 for num in degree_sequence):
            return 0

        # if the number of odd degree nodes are odd then it is not a graphic sequence
        num_odd = sum(1 for num in degree_sequence if num % 2 != 0)
        if num_odd % 2 == 1:
            return 0

        iteration = 1
        # Havel Hakimi algorithm code
        while degree_sequence:

            iteration_dict = {"degree sequence:": degree_sequence.copy()}

            # 1.Sort the degree sequence in decreasing order
            degree_sequence = sorted(degree_sequence, reverse=True)

            # 2.Check if all the elements are equal to zero it's a graphic sequence
            if all(num == 0 for num in degree_sequence):
                self.algorithm_log[f"Iteration {iteration}"] = iteration_dict
                return 1

            # 3.Store the first element in a variable and delete it from the sequence
            ele = degree_sequence[0]

            iteration_dict["highest node degree:"] = ele

            degree_sequence = degree_sequence[1:]
            iteration_dict['reduced degree sequence'] = degree_sequence.copy()

            # 4.If number of elements in the list are less than the first element then it's not a graphic sequence
            if len(degree_sequence) < ele:
                iteration_dict['is the sequence length smaller than highest degree ?'] = True
                self.algorithm_log[f"Iteration {iteration}"] = iteration_dict
                return 0
            else:
                iteration_dict['is the sequence length smaller than highest degree ?'] = False

            # 5.Subtract first element from next v elements
            i = 0
            for i in range(ele):
                degree_sequence[i] -= 1

            iteration_dict['modified degree sequence'] = degree_sequence.copy()

            # 6.Check if a negative element is encountered after subtraction,if yes return false
            if any(x < 0 for x in degree_sequence):
                iteration_dict['is there a negative element'] = True
                self.algorithm_log[f"Iteration {iteration}"] = iteration_dict
                return 0
            else:
                iteration_dict['is there a negative element'] = False

            self.algorithm_log[f"Iteration {iteration}"] = iteration_dict

            # 7.Repeat until either of the conditions is True
            iteration += 1

    @staticmethod
    def iteration_score(llm_iteration, gt_iteration):
        """
        This returns the score of every iteration by calculating the cumulative score of
        all the fields:
         highest node degree : redundant, same as degree_sequence[0]
                                This field will not be considered for evaluation
         reduced degree sequence : This field will be evaluated using our custom inversions score
         modified degree sequence : This field will be evaluated using our custom inversions score
         is there a negative element : boolean value (evaluation : 1 if same , 0 otherwise)
        """
        score_reduced_degree_sequence = inversion_score(llm_iteration['reduced degree sequence'],
                                                        gt_iteration['reduced degree sequence'])
        score_modified_degree_sequence = inversion_score(llm_iteration['modified degree sequence'],
                                                         gt_iteration['modified degree sequence'])
        score_is_there_negative_element = 1 if llm_iteration['is there a negative element'] == gt_iteration[
            'is there a negative element'] else 0

        iteration_score = 0.45 * score_reduced_degree_sequence + 0.45 * score_modified_degree_sequence + 0.1 * score_is_there_negative_element

        return iteration_score


class KuhnsAlgorithm(GraphAlgorithms):
    """Maximum matching in an unweighted bipartite graph using Kuhn's Algorithm"""

    def __init__(self, B):
        super().__init__()

        self.name = "Kuhns_Algorithm"
        self.task = "Find maximum matching in a bipartite graph"
        self.algorithm_steps = """
        The steps of Kuhn's Algorithm for a bipartite graph are as follows:
            For each node u in the left set:
                Initialize visited array for right nodes
                Call find_augmenting_path(u):
                    For each neighbor v of u:
                        If v is not visited:
                            Mark v visited
                            If v is unmatched or find_augmenting_path(match[v]):
                                Match v with u
                                Return True
            If an augmenting path is found:
                Add new edges to the maximum matching
        """
        self.schema = """
        {
            "maximum matching": list[tuple],
            "steps": {
                "Iteration [0-9]+$": {
                "current_left_node": integer,
                "visited_right_nodes": list[integer],
                "augmenting_path_found": boolean,
                "augmenting_path": list[tuple[integer,integer]],
                "matching_before": list[tuple[integer,integer]],
                "matching_after": list[tuple[integer,integer]],
            }
        """
        self.Graph = B
        self.left_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 0]
        self.right_nodes = [n for n, d in B.nodes(data=True) if d['bipartite'] == 1]
        self.matchR = {v: None for v in self.right_nodes}
        self.algorithm_log = {}

    def find_augmenting_path(self, u, visited, path):
        for v in self.Graph[u]:
            if v not in self.right_nodes:
                continue

            if visited[v]:
                continue

            visited[v] = True

            if self.matchR[v] is None or self.find_augmenting_path(self.matchR[v], visited, path):
                self.matchR[v] = u
                path.append((u, v))
                return True

        return False

    def current_matching(self):
        return [(u, v) for v, u in self.matchR.items() if u is not None]

    def kuhns_algorithm(self):
        iteration = 1

        for u in self.left_nodes:
            visited = {v: False for v in self.right_nodes}
            path = []

            matching_before = self.current_matching()
            found = self.find_augmenting_path(u, visited, path)

            self.algorithm_log[f"Iteration {iteration}"] = {
                "current_left_node": u,
                "visited_right_nodes": [x for x in visited if visited[x]],
                "augmenting_path_found": found,
                "augmenting_path": list(reversed(path)) if found else [],
                "matching_before": matching_before,
                "matching_after": self.current_matching()
            }

            iteration += 1

        return self.current_matching()

    def run(self):
        return self.kuhns_algorithm(), self.algorithm_log

    @staticmethod
    def iteration_score(llm_iteration, gt_iteration):
        pass


class KruskalsAlgorithm(GraphAlgorithms):
    """
    Implements Kruskal's Algorithm that find minimum spanning tree for undirected graphs
    """

    def __init__(self, G):
        super().__init__()

        self.name = "Kruskals_Algorithm"
        self.algorithm_steps = """
        KRUSKAL(G):
        G is a graph with vertices V and edges E
        Initialize MST as an empty set

        // 1. Create a disjoint set for each vertex
        for each vertex v that belongs to the vertex set of the graph G:
            Designate each vertex v as individual singleton sets

        // 2. Sort edges by weight in non-decreasing order
        sort the edges in the edge set of the Graph G in increasing order by weight of the edges (u, v)

        // 3. Process edges
        for each edge(u, v) that belongs to the sorted edge set of the graph G :
            if vertex u and vertex v do not belong to the same set then:
                Add edge (u, v) to MST
                Merge the sets to which vertex u and vertex v belong

        Return the MST
        """
        self.task = "Find the minimum spanning tree of the graph"
        self.schema = """
        {
            "minimum spanning tree": list[tuple],
            "steps": {
                "Iteration [0-9]+$": {
                "current edge":{'weight': int, 'u': int, 'v': int},
                "'current sorted edges list'": list[tuple[int],
                "parents before union":{'parent of u':int,'parent of v': int}
                'is parent of u and v same': boolean, 
                'edge accepted': boolean, 
                'MST':list[tuple[int]], 
                'MST Weight':int}
            }
        """
        self.Graph = G
        self.edges = []
        self.mst = []
        self.mst_weight = 0
        self.algorithm_log = {}

    def give_edges(self):
        adj_list = super().give_adjacency_list(self.Graph)
        edges = []
        for node in list(self.Graph.nodes):
            for adj in adj_list[node]:
                adj_node = adj[0]
                adj_node_wt = adj[1]
                edges.append([adj_node_wt, [node, adj_node]])
        self.edges = sorted(edges, key=lambda x: x[0])

    def kruskal_mst_algorithm(self):
        self.give_edges()
        disjoint_set = DisjointSet(self.Graph.nodes)
        self.mst_weight = 0

        iteration = 1

        for index, edge in enumerate(self.edges):

            iteration_dict = {}

            weight = edge[0]
            u = edge[1][0]
            v = edge[1][1]

            iteration_dict['current edge'] = {'weight': weight, 'u': u, 'v': v}
            iteration_dict['current sorted edges list'] = self.edges[index + 1:].copy()
            iteration_dict['parents before union'] = {'parent of u': disjoint_set.find_U_parent(u),
                                                      'parent of v': disjoint_set.find_U_parent(v)}
            # assuming initially the edge is not accepted , to be updated later if the edge is accepted
            iteration_dict.update(
                {'is parent of u and v same': True, 'edge accepted': False, 'MST': [], 'MST Weight': self.mst_weight})

            if disjoint_set.find_U_parent(u) != disjoint_set.find_U_parent(v):
                self.mst_weight += weight
                self.mst.append((u, v))
                disjoint_set.union_by_rank(u, v)

                # updating the iteration log to reflect the change in mst
                iteration_dict['is parent of u and v same'] = 'False'
                iteration_dict['edge accepted'] = True
                iteration_dict['MST'] = self.mst.copy()
                iteration_dict['MST Weight'] = self.mst_weight

            self.algorithm_log[f"Iteration {iteration}"] = iteration_dict
            iteration += 1

        return self.mst

    def run(self):
        return self.kruskal_mst_algorithm(), self.algorithm_log

    @staticmethod
    def iteration_score(llm_iteration, gt_iteration):
        pass
