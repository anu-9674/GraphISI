#  #this code extracts the prompts for graphs of num_nodes = 26 and saves them into a txt file

# import json
# import random
# from LLM_management import LLMManager,prompt_template,LLM_response
# from GraphAlgorithms import GraphAlgorithms
# from deterministic_graph_algorithms import BFSAlgorithm,DFSAlgorithm,DijkstraAlgorithm,HavelHakimiAlgorithm,KruskalsAlgorithm,KuhnsAlgorithm
# from graph_encoder import weighted_graph_encoder,unweighted_graph_encoder
# from networkx.readwrite import json_graph
# from ALGORITHM_CONFIG import is_weighted,is_directed

# #read the graph json file
# data=[]
# with open("Data/graphs.json") as f:
#     data=json.load(f)


# ALGORITHM_CONFIG = {
#               "bfs":{
#                 "class": BFSAlgorithm,
#                 "input_type": "graph",
#                 "generator_configurations": { "weighted":[False], "directed":[True,False],'bipartite':[False]},
#                 "encoder": unweighted_graph_encoder,
#             },
#             "dfs":{
#                 "class": DFSAlgorithm,
#                 "input_type": "graph",
#                 "generator_configurations": { "weighted": [False], "directed":[True,False],'bipartite':[False]},
#                 "encoder": unweighted_graph_encoder,
#             },
#             "dijkstra": {
#                 "class": DijkstraAlgorithm,
#                 "input_type": "graph",
#                 "generator_configurations": { "weighted": [True], "directed":[True,False],'bipartite':[False]},
#                 "encoder": weighted_graph_encoder,
#             },
#             "havel_hakimi": {
#                 "class": HavelHakimiAlgorithm,
#                 "input_type": "sequence",
#                 "generator_configurations": { "weighted": [False], "directed":[True,False],'bipartite':[False]},
#                 "encoder": lambda x: x,
#             },
#             "kuhn": {
#                 "class": KuhnsAlgorithm,
#                 "input_type": "graph",
#                 "generator_configurations": { "weighted":[False], "directed":[False],'bipartite':[True]},
#                 "encoder": unweighted_graph_encoder,
#             },
#             "kruskal": {
#                 "class": KruskalsAlgorithm,
#                 "input_type": "graph",
#                 "generator_configurations": { "weighted": [True], "directed":[False],'bipartite':[False] },
#                 "encoder": weighted_graph_encoder,
#             }
# }
# is_weighted = {True:'weighted',False:'unweighted'}
# is_directed = {True:'directed',False:'undirected'}

# class check:
#     def __init__(self,algorithm_name,model_cfg,k,top_p,top_k,temperature):
#         self.algorithm_name=algorithm_name
#         self.algorithm_class=ALGORITHM_CONFIG[algorithm_name]['class']
#         self.algorithm_object=self.algorithm_class(None)
#         self.input_type=ALGORITHM_CONFIG[algorithm_name]['input_type']
#         self.weighted=ALGORITHM_CONFIG[algorithm_name]['generator_configurations']['weighted']
#         self.directed=ALGORITHM_CONFIG[algorithm_name]['generator_configurations']['directed']
#         self.encoder=ALGORITHM_CONFIG[algorithm_name]['encoder']

#         self.k_shot=k
#         self.model_cfg=model_cfg
#         self.model_name=model_cfg['model_name']
#         self.top_p=top_p
#         self.top_k=top_k
#         self.temperature=temperature
#         self.llm_manager = LLMManager()
#         self.llm_manager.load_model(
#             model_name=model_cfg["model_name"],
#             model_id=model_cfg["model_id"],
#             context_length=model_cfg["context_window"] )

#     def in_context_learning_str(self,directed,weighted,nodes):
#         file_root = "examples_dataset"
#         file_path = f"{file_root}/{self.algorithm_name}_data_samples.json"
#         examples = GraphAlgorithms.select_random_samples(file_path, num_nodes=nodes,num_samples=self.k_shot,is_directed=directed,is_weighted=weighted)
#         # print("======================================")
#         # print(f"k_shot:"{self.k_shot}'')
#         # print("==================================================")
#         return GraphAlgorithms.create_example_string(examples, self.input_type)
        
        
#     def check_answer(self,encoded_input,weighted,directed,nodes):
#         """Check the performance of the model for low, mid and high node count graphs on the graph tasks"""
#         examples = self.in_context_learning_str(
#     directed=directed,
#     weighted=weighted,
#     nodes=nodes
# )
#         # print("=======================================================================")
#         # print(examples)
#         # print("=======================================================================")
#         prompt = prompt_template(examples, encoded_input,
#                                                     self.algorithm_object.get_task(),
#                                                     self.algorithm_object.get_algorithm_steps(), self.algorithm_object.get_schema(),
#                                                      self.input_type)
#         prompt=prompt.__str__()

        
#         return prompt
       
#     def run_check_answer(self):
#         prompts=[]

#         for node_count in data:
#             if(node_count== "26"):
#                 for weight in data[node_count]:
#                     for directed in data[node_count][weight]:
#                         for graph_json in data[node_count][weight][directed]:

#                             graph = json_graph.node_link_graph(graph_json)
#                             num_nodes=len(graph.nodes())

#                             if self.algorithm_name == "havel_hakimi":
#                                 encoded_input = HavelHakimiAlgorithm.give_degree_sequence(graph)
#                             else:
#                                 encoded_input = self.encoder(graph)

#                             algorithm_instance=self.algorithm_class(graph)
#                             #true_output ,log= algorithm_instance.run()    

#                             ans = self.check_answer(encoded_input,weighted=weight,directed=directed,nodes=num_nodes)
#                             prompts.append(ans)
#         with open("sample_prompts.json","w") as f:
#             json.dump(prompts,f)
#                     #print(f"NODE RANGE : {node_count}\n EDGE COUNT : {len(graph.edges)}\nWEIGHTED : {weight} \n DIRECTED : {directed}\n TRUE OUTPUT:{true_output}\n\n TRUE LOG : {log}\n\nANSWER: {ans}")
#                     #print("---------------------------------------------------------------------------------------------")
# model_cfg={
#             "model_name": "Gemma 3 12B",
#             "model_id": "google/gemma-3-12b-it",
#             "type": "Dense",
#             "context_window": 128000 
#         }
# check_dijsktra=check("bfs",model_cfg=model_cfg,k=0,top_p=0.5,top_k=50,temperature=0.5)
# check_dijsktra.run_check_answer()



'''trying the batch processing to see how well it works'''
import json
from more_itertools import batched
import LLM_management

def batch_llm_response_builder(inputs,llm_manager,model_name,top_p,top_k,temperature):
        """Takes in a batch of prompts and returns a dictionary containing the llm response to the prompt of a specified input id"""

        batch_llm_response=LLM_management.LLM_response(
                                   llm_manager=llm_manager,
                                   model_name=model_name,
                                   input_batch=inputs,
                                   top_p=top_p,
                                   top_k=top_k,
                                   temperature=temperature,
                                   batch=True)
        
        batch_output=dict.fromkeys([row[0] for row in inputs])
        index=0

        for input_id in batch_output.keys():
            batch_output[input_id]=batch_llm_response[index]
            index+=1

        return batch_output

def llm_response_builder(llm_manager,prompt,model_name,top_k,top_p,temperature):
        """Builds the prompt and returns the response from the LLM"""
        
        llm_output = LLM_management.LLM_response(llm_manager=llm_manager, 
                                                 prompt=prompt, 
                                                 model_name=model_name,
                                                 top_p=top_p,
                                                 top_k=top_k,
                                                 temperature=temperature)
        return llm_output

llm_manager = LLM_management.LLMManager()
model_cfg={
            "model_name": "Gemma 3 12B",
            "model_id": "google/gemma-3-12b-it",
            "type": "Dense",
            "context_window": 128000 
        }
llm_manager.load_model(
    model_name=model_cfg["model_name"],
    model_id=model_cfg["model_id"],
    context_length=model_cfg["context_window"] )

prompts=[]
with open("sample_prompts.json") as f:
    prompts=json.load(f)
for dict_batch in batched(prompts,4):
    generate_response=batch_llm_response_builder(inputs=dict_batch,model_name="Gemma 3 12B",llm_manager=llm_manager,top_p=1.0,top_k=50,temperature=0.7)

# for prompt in prompts:
#     generate_response=llm_response_builder(llm_manager=llm_manager,prompt=prompt,model_name="Gemma 3 12B",top_k=50,top_p=0.8,temperature=0.1)
#     print(generate_response)
