import json
import os
import random
import networkx as nx

"""Base class for all graph algorithm of all categories - deterministic,randomised and online"""
class GraphAlgorithms:
    def __init__(self):
        self.Graph = None
        self.algorithm_steps = ""
        self.task = ""
        self.name = ""
        self.schema = ""
        self.algorithm_log = {}

    def get_algorithm_steps(self):
        return self.algorithm_steps

    def get_task(self):
        return self.task

    def get_name(self):
        return self.name

    def get_schema(self):
        return self.schema

    def get_algorithm_log(self):
        return self.algorithm_log

    def run(self):
        """
        Runs and returns the result in the child classes
        """
    @staticmethod
    def select_random_samples(json_filepath,num_nodes,num_samples,is_weighted=None,is_directed=None):
        """Returns K examples to formulate k-shot prompting"""

        if num_samples==0:
                return []

        node_count_space={"low_node_range":[-2,-1,0,1,2],"mid_node_range":[-1,0,1],"high_node_range":[-3,-2,-1,0]}
        node_count_type = "low_node_range" if 3 <= int(num_nodes) <= 11 else "mid_node_range" if int(num_nodes) <= 21 else "high_node_range"

        try:
            with open(json_filepath, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print('Loaded json file data must be a list')
                return []
            if num_samples > len(data):
                print("Warning: Number of samples requested more than what is available")
                num_samples = len(data)

            data=data[0]

            sample_space=[]
            for i in node_count_space[node_count_type]:
                node_count=int(num_nodes)+i
                if is_weighted is not None :
                    sample_space.append(random.sample(data[node_count_type][str(node_count)][is_weighted][is_directed],1)[0])
                else:
                    sample_space.append(data[node_count_type])
            
            selected_random_samples=random.sample(sample_space,num_samples)

        except FileNotFoundError:
            print(f"Error: The file '{json_filepath}' was not found.")
            return []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the file '{json_filepath}'.")
            return []

        return selected_random_samples
    
    @staticmethod
    def write_to_file(json_filepath, data):
        """ Function to write into the file one by one as the samples come """

        if os.path.exists(json_filepath) and os.path.getsize(json_filepath) > 0:
            with open(json_filepath, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        existing_data.append(data)
        with open(json_filepath, "w") as f:
            json.dump(existing_data, f, indent=4)

    @staticmethod
    def give_adjacency_list(Graph):
        is_directed = Graph.is_directed()
        adjacency_list = {node: [] for node in Graph.nodes()}

        if nx.is_weighted(Graph):
            # Weighted graph
            for u, v, data in Graph.edges(data=True):
                weight = data.get("weight", 1)
                adjacency_list[u].append([v, weight])
                if not is_directed:
                    adjacency_list[v].append([u, weight])
        else:
            # Unweighted graph
            for u, v in Graph.edges():
                adjacency_list[u].append(v)
                if not is_directed:
                    adjacency_list[v].append(u)

        return adjacency_list
    
    @staticmethod
    def create_example_string(examples: list[dict],input_type) -> str:
        """Creates a string of the in_context learning examples to be appended to the prompt"""
    
        key_dict={'sequence':['input_sequence','input_sequence_result','input_sequence_algorithm_log'],
                'graph':['input_graph','input_graph_result','input_graph_algorithm_log']}
        key_dict=key_dict['sequence'] if input_type == 'sequence' else key_dict['graph']
        
        example_str = ''

        if len(examples) != 0:
            for cnt, example in enumerate(examples):
                input = example[key_dict[0]]
                result = example[key_dict[1]]
                steps = example[key_dict[2]]
                example_str += f'example:{cnt},{key_dict[0]}:{input}\n,result:{result}\n,steps:{steps}\n'

        return example_str

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class DisjointSet:
    """Implements the working of a Disjoint Set data structure
    This data structure has many applications in mst , randomised mst and karger's algorithm to be implemented
    later"""

    def __init__(self, V):
        self.rank = [0] * len(V)
        self.parent = list(range(len(V)))

    def union_by_rank(self, node1, node2):
        u_parent = self.find_U_parent(node1)
        v_parent = self.find_U_parent(node2)
        if u_parent == v_parent:
            return
        if self.rank[u_parent] < self.rank[v_parent]:
            self.parent[u_parent] = v_parent
        elif self.rank[u_parent] > self.rank[v_parent]:
            self.parent[v_parent] = u_parent
        else:
            self.parent[v_parent] = u_parent
            self.rank[u_parent] += 1

    def find_U_parent(self, node):
        """Helps us check to which component a particular node belongs to"""
        if self.parent[node] != node:
            self.parent[node] = self.find_U_parent(self.parent[node])
        return self.parent[node]
