## Steps to work with the code
1. Create a read token from huggingFace to access the gated models and paste in the LLM_management code.
2. Run file run.py to create all necessary folder and files
3. Optinal : Run file LLM_Management.py to download the required model(other than gemma3).
4. To build the dataset run the main.py code with the arguments like this : 

python main.py \
  --algorithm_type deterministic \
  --algorithm bfs \
  --k 1 \
  --model_type Simple \
  --model_name gemma-3-12b-it \
  --temperature 0.1 \
  --top_p 0.1 \
  --top_k 10 
