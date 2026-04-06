## Steps to work with the code
1. Run file run.py to create all necessary folder and files

2.To build the dataset run the main.py code with the arguments like this : 

python main.py \
  --algorithm_type deterministic \
  --algorithm bfs \
  --k 1 \
  --model_type Simple \
  --model_name gemma-3-12b-it \
  --temperature 0.1 \
  --top_p 0.1 \
  --top_k 10 
