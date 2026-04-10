"""
This file generates the query input file for all the algorithms
The query input file contains the query graph encoded using text and the ground truth,
where the ground truth contains the actual result and the steps of the algorithm required to reach that result.
"""

from deterministic_graph_algorithms import BFSAlgorithm,DFSAlgorithm,DijkstraAlgorithm,HavelHakimiAlgorithm,KruskalsAlgorithm,KuhnsAlgorithm
from graph_encoder import unweighted_graph_encoder,weighted_graph_encoder
from GraphAlgorithms import GraphAlgorithms
import numpy as np
import os 
import json
from networkx.readwrite import json_graph
from collections import defaultdict
from ALGORITHM_CONFIG import ALGORITHM_CONFIG, is_weighted, is_directed
from pathlib import Path

Path("Output").mkdir(exist_ok=True)



data_id=1

class QueryInputBuilder:

    def __init__(self,algorithm,algorithm_type,k_shot=0):

        Path(f"Output/{algorithm}").mkdir(exist_ok=True)

        self.graph_file_path="Data/graphs.json"
        self.bipartite_graph_file="Data/bipartite_graphs.json"

        self.algorithm_name=algorithm
        self.config = ALGORITHM_CONFIG[algorithm_type][self.algorithm_name]
        self.algorithm = self.config['class']
        self.encoder_function = self.config['encoder']
        self.input_type = self.config['input_type']

        self.graph_file_path= self.bipartite_graph_file if self.algorithm_name == 'kuhn' or self.algorithm_name=='online_bipartite' else self.graph_file_path
        self.query_input_filepath=f"Output/{self.algorithm_name}/{self.algorithm_name}_query_inputs.json"
        self.output_filepath=self.output_file = f"Dataset/{self.algorithm_name}_{k_shot}_.json"

        Path(self.query_input_filepath).touch(exist_ok=True)
    
    def query_input_builder(self,query_input):
        """Builds the query input file by keeping record of the query input and the 
            corresponding result and the algorithm steps"""

        algorithm_instance = self.algorithm(query_input)
        query_input_result, query_input_algorithm_log = algorithm_instance.run()

        global data_id
      
        data = {'data id': data_id}

        if self.input_type == 'sequence':
                data['sequence_lenght']=len(query_input)
        else:
                data['query_graph size']= {'nodes': len(query_input.nodes), 'edges': len(query_input.edges)}
        data.update({"query_input": self.encoder_function(query_input),
                    'query_input_result': query_input_result,
                    'query_input_algorithm_log': query_input_algorithm_log})
        data_id+=1
        return data
    
    def read_graphs_for_query_graphs(self):       
        """Reads the graph from the dataset and passes them onto the query builder function"""

        graphs_list = dict({})
        query_input_results={str(num_nodes): {} for num_nodes in range(5,31)}
        
        with open(self.graph_file_path) as f:
                graphs_list = json.load(f)
          
        generator_configurations=self.config["generator_configurations"]
        weighted_configurations=generator_configurations['weighted']
        directed_configurations=generator_configurations['directed']

        for num_nodes in graphs_list:
            result = {
        "weighted":   {"directed": [], "undirected": []},
        "unweighted": {"directed": [], "undirected": []},
        }
            for weighted_configuration in weighted_configurations:
                    for directed_configuration in directed_configurations:
                        graphs=graphs_list[num_nodes][is_weighted[weighted_configuration]][is_directed[directed_configuration]]
                        for graph in graphs:
                                data = self.query_input_builder(json_graph.node_link_graph(graph))
                                result[is_weighted[weighted_configuration]][is_directed[directed_configuration]].append(data)
            query_input_results[num_nodes]=result
        os.makedirs(os.path.dirname(self.graph_file_path), exist_ok=True)
        with open(self.query_input_filepath, "w") as f:
            json.dump(query_input_results, f, indent=4)
    

    def read_graphs_for_query_sequence(self):
        """Reads graphs from the dataset then prepares sequences from it"""

        graphs_list={}
        with open(self.graph_file_path) as f:
                graphs_list = json.load(f)

        weighted_configurations=self.config["generator_configurations"]['weighted']
        directed_configurations=self.config["generator_configurations"]['directed']
        
        query_input_results={str(n):[] for n in range(5,31)}

        for num_nodes in graphs_list:
            sequences=[]
            for weighted_configuration in weighted_configurations:
                  for directed_configuration in directed_configurations:
                       graphs=graphs_list[num_nodes][is_weighted[weighted_configuration]][is_directed[directed_configuration]]
                       for graph in graphs:
                                sequence = self.query_input_builder(HavelHakimiAlgorithm.give_degree_sequence(json_graph.node_link_graph(graph)))
                                sequences.append(sequence)
            query_input_results[num_nodes]=sequences

        os.makedirs(os.path.dirname(self.graph_file_path), exist_ok=True)
        with open(self.query_input_filepath, "w") as f:
            json.dump(query_input_results, f, indent=4)      


    def run_read_graphs(self):
         if self.algorithm_name=='havel_hakimi':
              self.read_graphs_for_query_sequence()
         else:
              self.read_graphs_for_query_graphs()  
    

    def read_graphs_for_output_graphs(self):
        data_id=1
        graphs_list = dict({})
        output_results={str(num_nodes): {} for num_nodes in range(5,31)}

        generator_configurations=self.config["generator_configurations"]
        weighted_configurations=generator_configurations['weighted']
        directed_configurations=generator_configurations['directed']

        with open(self.graph_file_path) as f:
                graphs_list = json.load(f)
        for num_nodes in graphs_list:
            result = {
        "weighted":   {"directed": {'input':{}}, "undirected":{'input':{}} },
        "unweighted": {"directed":{'input':{}}, "undirected":{'input':{}}},
        }
            for weighted_configuration in weighted_configurations:
                    for directed_configuration in directed_configurations:
                        graphs=graphs_list[num_nodes][is_weighted[weighted_configuration]][is_directed[directed_configuration]]
                        for graph in graphs:
                                data = self.query_input_builder(json_graph.node_link_graph(graph))
                                result[is_weighted[weighted_configuration]][is_directed[directed_configuration]]['input'][data_id]=data['query_input']
                                data_id+=1
            output_results[num_nodes]=result
        with open(self.output_filepath, "w") as f:
            json.dump(output_results, f, indent=4)
    
    def read_graphs_for_output_sequences(self):
        data_id=1
        graphs_list = dict({})
        output_results={str(num_nodes): {} for num_nodes in range(5,31)}

        generator_configurations=self.config["generator_configurations"]
        weighted_configurations=generator_configurations['weighted']
        directed_configurations=generator_configurations['directed']

        with open(self.graph_file_path) as f:
                graphs_list = json.load(f)
        for num_nodes in graphs_list:
            recursive_dict = lambda: defaultdict(recursive_dict)
            result = recursive_dict()
            for weighted_configuration in weighted_configurations:
                    for directed_configuration in directed_configurations:
                        graphs=graphs_list[num_nodes][is_weighted[weighted_configuration]][is_directed[directed_configuration]]
                        for graph in graphs:
                                data = self.query_input_builder(HavelHakimiAlgorithm.give_degree_sequence(json_graph.node_link_graph(graph)))
                                result['input'][data_id]=data['query_input']
                                data_id+=1
            output_results[num_nodes]=result
        with open(self.output_filepath, "w") as f:
            json.dump(output_results, f, indent=4)
    
    def read_graphs_for_output(self):
          if self.algorithm_name=='havel_hakimi':
                self.read_graphs_for_output_sequences()
          else:
                self.read_graphs_for_output_graphs()
              
def main(args=None):
    """Main fucntion to run the file builder class"""
    algorithms={"deterministic":["bfs","dfs","dijkstra","havel_hakimi","kuhn","kruskal"]}
    for algorithm_type in algorithms:
        for algorithm in algorithms[algorithm_type]:
            query_obj=QueryInputBuilder(algorithm,algorithm_type,0)
            query_obj.run_read_graphs()
            query_obj.read_graphs_for_output()

            query_obj=QueryInputBuilder(algorithm,algorithm_type,1)
            query_obj.read_graphs_for_output()

            query_obj=QueryInputBuilder(algorithm,algorithm_type,2)
            query_obj.read_graphs_for_output()

if __name__ == "__main__":
    main()
                

