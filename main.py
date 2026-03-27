import json
import argparse
import LLM_management
from DatasetBuilder import DatasetBuilder
with open('model_registry.json', 'r') as f:
    MODEL_REGISTRY = json.load(f)

def main(algorithm_type,algorithm_name,model_cfg,k_value,temperature,top_p,top_k):
    """Runs the DatasetBuilder code from the parsed arguments"""

    object=DatasetBuilder(algorithm_name=algorithm_name,
                          algorithm_type=algorithm_type,
                          model_cfg=model_cfg,
                          k_shot=k_value,
                          temperature=temperature,
                          top_p=top_p,
                          top_k=top_k                   
                          )
    print(f"Generating output data for {algorithm_name} with model {model_cfg['model_name']}")
    print("-----------------------------------------------------------------------")
    object.run()


if __name__ == "__main__":

    """Parses the command line arguments and then runs the dataset generator accordingly"""

    parser = argparse.ArgumentParser(description="Generate k-shot ICL datasets for given graph algorithm")

    parser.add_argument(
        "--algorithm_type",
        type=str,
        required=True,
        choices=['deterministic','randomised','online'],
        help="Type of algorithm to generate data for",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["dijkstra", "havel_hakimi", "kuhn", "kruskal",'karger','randomised_mst','online'],
        help="Algorithm to generate data for"
    )
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        default=0,
        choices=[0, 1, 2],
        help="Number of in-context examples for k-shot queries"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['Advanced', 'Simple'],
        help="Type of large model to be used"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of large model to be used for study"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help='Sets the temperature that controls the creativity of the model response'
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.1,
        help="Sets the value of top_p used in generating the response"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help='Sets the value of top_k used in generating the model response'
    )
    
    args = parser.parse_args()

    if args.model_name not in MODEL_REGISTRY[args.model_type]:
        available_models = list(MODEL_REGISTRY[args.model_type].keys())
        raise ValueError(
            f"Model Unavailable '{args.model_name}' for type '{args.model_type}'. "
            f"Valid options: {available_models}")

    model_cfg = MODEL_REGISTRY[args.model_type][args.model_name]
    
    main  ( algorithm_type=args.algorithm_type,
            algorithm_name=args.algorithm,
            model_cfg=model_cfg,
            k_value=args.k,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k)
    
