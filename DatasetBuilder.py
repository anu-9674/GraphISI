"""
We will be preparing examples from different algorithms to use during in context learning , by picking up
K examples from the file for k-shot prompting
"""
import os
import json
import LLM_management
from collections import defaultdict
from GraphAlgorithms import GraphAlgorithms
from ALGORITHM_CONFIG import ALGORITHM_CONFIG , is_weighted,is_directed
from tqdm import tqdm
from GraphAlgorithms import GraphAlgorithms

with open('model_registry.json', 'r') as f:
    MODEL_REGISTRY = json.load(f)

class DatasetBuilder:

    def __init__(self,algorithm_type,algorithm_name,k_shot,model_cfg,temperature,top_p,top_k):

        # self.output_file = f"Dataset/{algorithm_name}/{algorithm_name}_{k_shot}_.json"

        self.output_file = f"Output/{algorithm_name}/{algorithm_name}_{k_shot}_.json"

        self.algorithm_name=algorithm_name
        self.algorithm_type=algorithm_type
        self.algorithm_object = ALGORITHM_CONFIG[self.algorithm_type][algorithm_name]['class'](nx.Graph())
        self.weighted_configurations=ALGORITHM_CONFIG[self.algorithm_type][self.algorithm_name]['generator_configurations']['weighted']
        self.directed_configurations=ALGORITHM_CONFIG[self.algorithm_type][self.algorithm_name]['generator_configurations']['directed']
        self.input_type=ALGORITHM_CONFIG[self.algorithm_type][algorithm_name]['input_type']

        self.k_shot = k_shot
        self.model_name = model_cfg['model_name']

        #loading the model
        self.llm_manager = LLM_management.LLMManager()
        self.llm_manager.load_model(
            model_name=model_cfg["model_name"],
            model_id=model_cfg["model_id"],
            context_length=model_cfg["context_window"] )
        #model configurations
        self.temperature=temperature
        self.top_p=top_p 
        self.top_k=top_k


    def in_context_learning_examples(self,num_nodes,is_weighted=None,is_directed=None):##
        """Returns the in-context learing examples in proper prompt format"""

        file_root = "examples_dataset"
        file_path = f"{file_root}/{self.algorithm_name}_data_samples.json"
        examples = GraphAlgorithms.select_random_samples(file_path,
                                                        num_nodes=num_nodes,
                                                        num_samples=self.k_shot,
                                                        is_weighted=is_weighted,
                                                        is_directed=is_directed)

        in_context_example_str = GraphAlgorithms.create_example_string(examples, ALGORITHM_CONFIG[self.algorithm_type][self.algorithm_name][
            'input_type'])
        
        return in_context_example_str
    

    def llm_response_builder(self, encoded_query_input:str, in_context_example_str:str, input_type:str, algorithm_object:GraphAlgorithms,
                             model_name:str):
        """Builds the prompt and returns the response from the LLM"""

        prompt = LLM_management.prompt_template(in_context_example_str, encoded_query_input,
                                                algorithm_object.get_task(),
                                                algorithm_object.get_algorithm_steps(), algorithm_object.get_schema(),
                                                input_type)
        
        llm_output = LLM_management.LLM_response(self.llm_manager, 
                                                 prompt, 
                                                 model_name,
                                                 top_p=self.top_p,
                                                 top_k=self.top_k,
                                                 temperature=self.temperature)

        return llm_output


    def write_to_output_file_for_graphs(self):
        """Read the inputs from the file and then records the model responses"""

        data={}
        with open(self.output_file,"r") as f:
            data=json.load(f)
        
        # for num_nodes in tqdm(data, desc=f"Processing {self.algorithm_name}"):
        for cnt, num_nodes in enumerate(data):
            print(f"------Processing {self.algorithm_name} at {cnt+1}/{len(data)}-----")
            for weighted in self.weighted_configurations:
                w_key=is_weighted[weighted]
                for directed in self.directed_configurations:
                    d_key=is_directed[directed]

                    if self.model_name not in data[num_nodes][w_key][d_key]:
                        data[num_nodes][w_key][d_key][self.model_name] = {}
           
                    input_data = data[num_nodes][w_key][d_key]['input']

                    for input_index, query in tqdm(input_data.items(), desc="Queries", leave=False):
                        llm_response = self.llm_response_builder(
                            encoded_query_input=query,
                            in_context_example_str=self.in_context_learning_examples(num_nodes=num_nodes,
                                                                                    is_weighted=w_key,
                                                                                    is_directed=d_key),
                            input_type=self.input_type,
                            algorithm_object=self.algorithm_object,
                            model_name=self.model_name
                        )
                        data[num_nodes][w_key][d_key][self.model_name][input_index] = llm_response
            if cnt%10 == 0 or cnt+1 == len(data):    
                with open(f"{self.output_file}_{self.model_name}","w") as f:
                    json.dump(data,f)

    def write_to_output_file_for_sequences(self):
        data={}
        with open(self.output_file,"r") as f:
            data=json.load(f)
        
        # for num_nodes in tqdm(data, desc=f"Processing {self.algorithm_name}"):
        for cnt, num_nodes in enumerate(data):
                    if self.model_name not in data[num_nodes]:
                        data[num_nodes][self.model_name] = {}
           
                    input_data = data[num_nodes]['input']

                    for input_index, query in tqdm(input_data.items(), desc="Queries", leave=False):
                        llm_response = self.llm_response_builder(
                            encoded_query_input=query,
                            in_context_example_str=self.in_context_learning_examples(num_nodes=num_nodes,),
                            input_type=self.input_type,
                            algorithm_object=self.algorithm_object,
                            model_name=self.model_name
                        )
                        data[num_nodes][self.model_name][input_index] = llm_response
                    if cnt%10 == 0 or cnt+1 == len(data):    
                        with open(f"Output/{self.algorithm_name}/{self.algorithm_name}_{self.k_shot}_{self.model_name}.json","w") as f:
                            json.dump(data,f)

    
    def run(self):
        if self.algorithm_name!= 'havel_hakimi':
            self.write_to_output_file_for_graphs()
        else :
             self.write_to_output_file_for_sequences()


