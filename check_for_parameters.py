"""
This code just prints a response of an llm in k-shot setting for a particular algorithm for graphs in the 
low , mid and high node count range.This will help us to keep track of the parameters to be used in different scenarios.

We will choose a graph from the low, mid and high count range and run them with all the algorithm in different k shot settings
"""
import json
import random
from LLM_management import LLMManager,prompt_template,LLM_response
from GraphAlgorithms import GraphAlgorithms
from deterministic_graph_algorithms import BFSAlgorithm,DFSAlgorithm,DijkstraAlgorithm,HavelHakimiAlgorithm,KruskalsAlgorithm,KuhnsAlgorithm
from graph_encoder import weighted_graph_encoder,unweighted_graph_encoder
from networkx.readwrite import json_graph
from ALGORITHM_CONFIG import is_weighted,is_directed

#read the graph json file
data=[]
with open("Data/graphs.json") as f:
    data=json.load(f)
with open("Data/bipartite_graphs.json") as f:
    bipartite_data=json.load(f)

graphs = {"low_node_count":{
    "weighted":{
        "directed":random.sample(data["6"]['weighted']['directed'],1)[0],
        "undirected":random.sample(data["6"]['weighted']['undirected'],1)[0]
    },
    "unweighted":{
        "directed":random.sample(data["6"]['unweighted']['directed'],1)[0],
        "undirected":random.sample(data["6"]['unweighted']['undirected'],1)[0],
    }
    },
        "mid_node_count":{
            "weighted":{
                "directed":random.sample(data["15"]['weighted']['directed'],1)[0],
                "undirected":random.sample(data["15"]['weighted']['undirected'],1)[0],
            },
            "unweighted":{
                "directed":random.sample(data["15"]['unweighted']['directed'],1)[0],
                "undirected":random.sample(data["15"]['unweighted']['undirected'],1)[0],
            }
        },

        "high_node_count":{
        "weighted":{
            "directed":random.sample(data["27"]['weighted']['directed'],1)[0],
            "undirected":random.sample(data["27"]['weighted']['undirected'],1)[0],
        },
        "unweighted":{
            "directed":random.sample(data["27"]['unweighted']['directed'],1)[0],
            "undirected":random.sample(data["27"]['unweighted']['undirected'],1)[0],
        }
        },
        "bipartite":{"low_node_count":{
    "weighted":{
        "directed":random.sample(bipartite_data["6"]['weighted']['directed'],1)[0],
        "undirected":random.sample(bipartite_data["6"]['weighted']['undirected'],1)[0]
    },
    "unweighted":{
        "directed":random.sample(bipartite_data["6"]['unweighted']['directed'],1)[0],
        "undirected":random.sample(bipartite_data["6"]['unweighted']['undirected'],1)[0],
    }
    },
        "mid_node_count":{
            "weighted":{
                "directed":random.sample(bipartite_data["15"]['weighted']['directed'],1)[0],
                "undirected":random.sample(bipartite_data["15"]['weighted']['undirected'],1)[0],
            },
            "unweighted":{
                "directed":random.sample(bipartite_data["15"]['unweighted']['directed'],1)[0],
                "undirected":random.sample(bipartite_data["15"]['unweighted']['undirected'],1)[0],
            }
        },

        "high_node_count":{
        "weighted":{
            "directed":random.sample(bipartite_data["27"]['weighted']['directed'],1)[0],
            "undirected":random.sample(bipartite_data["27"]['weighted']['undirected'],1)[0],
        },
        "unweighted":{
            "directed":random.sample(bipartite_data["27"]['unweighted']['directed'],1)[0],
            "undirected":random.sample(bipartite_data["27"]['unweighted']['undirected'],1)[0],
        },

    },
}
}


ALGORITHM_CONFIG = {
              "bfs":{
                "class": BFSAlgorithm,
                "input_type": "graph",
                "generator_configurations": { "weighted":[False], "directed":[True,False],'bipartite':[False]},
                "encoder": unweighted_graph_encoder,
            },
            "dfs":{
                "class": DFSAlgorithm,
                "input_type": "graph",
                "generator_configurations": { "weighted": [False], "directed":[True,False],'bipartite':[False]},
                "encoder": unweighted_graph_encoder,
            },
            "dijkstra": {
                "class": DijkstraAlgorithm,
                "input_type": "graph",
                "generator_configurations": { "weighted": [True], "directed":[True,False],'bipartite':[False]},
                "encoder": weighted_graph_encoder,
            },
            "havel_hakimi": {
                "class": HavelHakimiAlgorithm,
                "input_type": "sequence",
                "generator_configurations": { "weighted": [False], "directed":[True,False],'bipartite':[False]},
                "encoder": lambda x: x,
            },
            "kuhn": {
                "class": KuhnsAlgorithm,
                "input_type": "graph",
                "generator_configurations": { "weighted":[False], "directed":[False],'bipartite':[True]},
                "encoder": unweighted_graph_encoder,
            },
            "kruskal": {
                "class": KruskalsAlgorithm,
                "input_type": "graph",
                "generator_configurations": { "weighted": [True], "directed":[False],'bipartite':[False] },
                "encoder": weighted_graph_encoder,
            }
}
is_weighted = {True:'weighted',False:'unweighted'}
is_directed = {True:'directed',False:'undirected'}

class check:
    def __init__(self,algorithm_name,model_cfg,k,top_p,top_k,temperature):
        self.algorithm_name=algorithm_name
        self.algorithm_class=ALGORITHM_CONFIG[algorithm_name]['class']
        self.algorithm_object=self.algorithm_class(None)
        self.input_type=ALGORITHM_CONFIG[algorithm_name]['input_type']
        self.weighted=ALGORITHM_CONFIG[algorithm_name]['generator_configurations']['weighted']
        self.directed=ALGORITHM_CONFIG[algorithm_name]['generator_configurations']['directed']
        self.encoder=ALGORITHM_CONFIG[algorithm_name]['encoder']

        self.k_shot=k
        self.model_cfg=model_cfg
        self.model_name=model_cfg['model_name']
        self.top_p=top_p
        self.top_k=top_k
        self.temperature=temperature
        self.llm_manager = LLMManager()
        self.llm_manager.load_model(
            model_name=model_cfg["model_name"],
            model_id=model_cfg["model_id"],
            context_length=model_cfg["context_window"] )

    def in_context_learning_str(self,directed,weighted,nodes):
        file_root = "examples_dataset"
        file_path = f"{file_root}/{self.algorithm_name}_data_samples.json"
        examples = GraphAlgorithms.select_random_samples(file_path, num_nodes=nodes,num_samples=self.k_shot,is_directed=directed,is_weighted=weighted)
        # print("======================================")
        # print(f"k_shot:"{self.k_shot}'')
        # print("==================================================")
        return GraphAlgorithms.create_example_string(examples, self.input_type)
        
        
    def check_answer(self,encoded_input,weighted,directed,nodes):
        """Check the performance of the model for low, mid and high node count graphs on the graph tasks"""
        examples = self.in_context_learning_str(
    directed=directed,
    weighted=weighted,
    nodes=nodes
)
        # print("=======================================================================")
        # print(examples)
        # print("=======================================================================")
        prompt = prompt_template(examples, encoded_input,
                                                    self.algorithm_object.get_task(),
                                                    self.algorithm_object.get_algorithm_steps(), self.algorithm_object.get_schema(),
                                                    self.input_type)
        
        llm_output = LLM_response(self.llm_manager, prompt, self.model_name,top_p=self.top_p,top_k=self.top_k,temperature=self.temperature)

        return llm_output
       
    def run_check_answer(self):
        if self.algorithm_name == "kuhn":
            graph_dict = graphs["bipartite"]
        else:
            graph_dict = {k:v for k,v in graphs.items() if k!="bipartite"}
        for node_count, node_data in graph_dict.items():
            for weight in self.weighted:
                for directed in self.directed:

                    graph_json = node_data[is_weighted[weight]][is_directed[directed]]

                    graph = json_graph.node_link_graph(graph_json)
                    num_nodes=len(graph.nodes())

                    if self.algorithm_name == "havel_hakimi":
                        encoded_input = HavelHakimiAlgorithm.give_degree_sequence(graph)
                    else:
                        encoded_input = self.encoder(graph)

                    algorithm_instance=self.algorithm_class(graph)
                    true_output ,log= algorithm_instance.run()    

                    ans = self.check_answer(encoded_input,weighted=is_weighted[weight],directed=is_directed[directed],nodes=num_nodes)
                    print(f"NODE RANGE : {node_count}\n EDGE COUNT : {len(graph.edges)}\nWEIGHTED : {weight} \n DIRECTED : {directed}\n TRUE OUTPUT:{true_output}\n\n TRUE LOG : {log}\n\nANSWER: {ans}")
                    print("---------------------------------------------------------------------------------------------")
model_cfg={
            "model_name": "Gemma 3 12B",
            "model_id": "google/gemma-3-12b-it",
            "type": "Dense",
            "context_window": 128000 
        }
check_dijsktra=check("bfs",model_cfg=model_cfg,k=0,top_p=0.5,top_k=50,temperature=0.5)
check_dijsktra.run_check_answer()

    
