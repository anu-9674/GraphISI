"""
In this file we will be generating data from the algorithms to write into their respective files
To do the ablation study for in-context learning we will be sampling from these files
"""
import json
import os
import numpy as np
from collections import defaultdict

from pathlib import Path
Path("examples_dataset").mkdir(exist_ok=True)

from graph_encoder import weighted_graph_encoder, unweighted_graph_encoder
import graph_generators
from deterministic_graph_algorithms import BFSAlgorithm,DFSAlgorithm,DijkstraAlgorithm, HavelHakimiAlgorithm, \
    KuhnsAlgorithm, KruskalsAlgorithm

example_id = 1
weighted={True:'weighted',False:'unweighted'}
directed={True:'directed',False:'undirected'}


class FileBuilder:

    def __init__(self, algorithm_name):
        self.algorithm = algorithm_name
        self.ALGORITHM_CONFIG = {
            "bfs":{
                "class": BFSAlgorithm,
                "input_type": "graph",
                "graph_generators": [
                    {"algorithm": "er", "weighted": False, "directed": False},
                    {"algorithm": "er", "weighted": False, "directed": True},
                ],
                "encoder": unweighted_graph_encoder,
            },
            "dfs":{
                "class": DFSAlgorithm,
                "input_type": "graph",
                "graph_generators": [
                    {"algorithm": "er", "weighted": False, "directed": False},
                    {"algorithm": "er", "weighted": False, "directed": True},
                ],
                "encoder": unweighted_graph_encoder,
            },
            "dijkstra": {
                "class": DijkstraAlgorithm,
                "input_type": "graph",
                "graph_generators": [
                    {"algorithm": "er", "weighted": True, "directed": False},
                    {"algorithm": "er", "weighted": True, "directed": True}

                ],
                "encoder": weighted_graph_encoder,
            },
            "havel_hakimi": {
                "class": HavelHakimiAlgorithm,
                "input_type": "sequence",
                "sequence_generators": [
                    {"algorithm": "er", "weighted": False, "directed": False},
                    {"algorithm": "er", "weighted": False, "directed": True},
                ],
                "encoder": lambda x: x,
            },
            "kuhn": {
                "class": KuhnsAlgorithm,
                "input_type": "graph",
                "graph_generators": [
                    {"algorithm": "bipartite", "weighted": False, "directed": False}],
                "encoder": unweighted_graph_encoder,
            },
            "kruskal": {
                "class": KruskalsAlgorithm,
                "input_type": "graph",
                "graph_generators": [
                    {"algorithm": "er", "weighted": True, "directed": False},
                ],
                "encoder": weighted_graph_encoder,
            }
        }
        node_sizes={"low_node_range":np.arange(3,12).tolist(),"mid_node_range":np.arange(12,22).tolist(),"high_node_range":np.arange(22,31).tolist()}
        self.GRAPH_SIZES = node_sizes
        self.filepath = f"examples_dataset/{self.algorithm}_data_samples.json"
        #create file if it does not exist
        Path(self.filepath).touch(exist_ok=True)


    def generate_data(self):
        config = self.ALGORITHM_CONFIG[self.algorithm]
        global example_id

        recursive_dict = lambda: defaultdict(recursive_dict)
        results = recursive_dict()

        algorithm = config['class']
        graph_encoder_function = config['encoder']
        input_type = config['input_type']
        generators = config['graph_generators'] if input_type == 'graph' else config['sequence_generators']

        for graph_size_range in self.GRAPH_SIZES:
            
            for graph_size in self.GRAPH_SIZES[graph_size_range]:
                for generator in generators:
                    #for each configuration we will keep 4 examples
                    samples=[]
                    for i in range(4):
                        gen_algorithm, is_weighted, is_directed = generator['algorithm'], generator['weighted'], generator[
                            'directed']
                        
                        data_input = graph_generators.generate_graphs(1, gen_algorithm, is_weighted, is_directed,number_of_nodes=graph_size)[0]

                        algorithm_instance = algorithm(data_input)
                        data = {'id': example_id, 'Algorithm': algorithm_instance.get_name()}

                        input_result, input_algorithm_log = algorithm_instance.run()

                        data.update({'input_graph': graph_encoder_function(data_input),
                                        'input_graph size': {'nodes': len(data_input.nodes), 'edges': len(data_input.edges)},
                                        'input_graph_result': input_result,
                                        'input_graph_algorithm_log': input_algorithm_log}) 
                        samples.append(data)
                        example_id+=1

                    results[graph_size_range][graph_size][weighted[is_weighted]][directed[is_directed]]=samples

                        
        self.write_to_file(results)

    def generate_sequence_data(self):
        config = self.ALGORITHM_CONFIG[self.algorithm]
        global example_id

        recursive_dict = lambda: defaultdict(recursive_dict)
        results = recursive_dict()
        
        algorithm = config['class']
        input_type = config['input_type']
        generators = config['graph_generators'] if input_type == 'graph' else config['sequence_generators']

        for graph_size_range in self.GRAPH_SIZES:
            
            for graph_size in self.GRAPH_SIZES[graph_size_range]:
                for generator in generators:
                    #for each configuration we will keep 4 examples
                    samples=[]
                    for i in range(4):
                        gen_algorithm, is_weighted, is_directed = generator['algorithm'], generator['weighted'], generator[
                            'directed']

                        data_input = graph_generators.generate_sequences(1, gen_algorithm, is_weighted, is_directed,number_of_nodes=graph_size)[0]
                        algorithm_instance = algorithm(data_input)
                        data = {'id': example_id, 'Algorithm': algorithm_instance.get_name()}

                        input_result, input_algorithm_log = algorithm_instance.run()

                        data.update({'input_sequence': data_input, 'sequence length': len(data_input),
                                        'input_sequence_result': input_result,
                                        'input_sequence_algorithm_log': input_algorithm_log})
                        samples.append(data)
                        example_id+=1

                    results[graph_size_range][graph_size]=samples
        
        self.write_to_file(results)

    def write_to_file(self, data):
        """ Function to write into the file one by one as the samples come"""

        if os.path.exists(self.filepath) and os.path.getsize(self.filepath) > 0:
            with open(self.filepath, "r") as f:
                existing_data = json.load(f)
        else:
            existing_data = []
        existing_data.append(data)
        with open(self.filepath, "w") as f:
            json.dump(existing_data, f, indent=4)
    
    def run(self):
        if self.algorithm == 'havel_hakimi':
            self.generate_sequence_data()
        else:
            self.generate_data()

def main(args=None):
    """Main fucntion to run the file builder class"""
    algorithms=["bfs","dfs","dijkstra","havel_hakimi","kuhn","kruskal"]
    print("Generating samples for in-context learning..")
    for algorithm in algorithms:
        fileBuilder_obj=FileBuilder(algorithm)
        fileBuilder_obj.run()
        print(f"Saved in_context learning examples for {algorithm} algorithm")

if __name__ == "__main__":
    main()
