1.Create the graphs and the bipartite graphs dataset by running the graph_generator.py code

2.Generate the in_context learning dataset by running the in_context_learning_examples.py code for that particular algorithm for example ,bfs.

3.Generate the query file which contains the encoded inputs and the corresponding ground truth by running the file read_graphs function in Query_input_builder.py for the same algorithm as above

4.Generate the k_json file by running the Query_input_builder.py file 

5.To build the dataset run the main.py code with the arguments like this : python main.py \
  --algorithm_type deterministic \
  --algorithm bfs \
  --k 1 \
  --model_type Simple \
  --model_name gemma-3-12b-it \
  --temperature 0.1 \
  --top_p 0.1 \
  --top_k 10 
