from deterministic_graph_algorithms import BFSAlgorithm,DFSAlgorithm,DijkstraAlgorithm,HavelHakimiAlgorithm,KruskalsAlgorithm,KuhnsAlgorithm
from randomised_graph_algorithms import KargerAlgorithm,RandomisedMST
from graph_encoder import weighted_graph_encoder,unweighted_graph_encoder
import numpy as np

ALGORITHM_CONFIG={"deterministic":{
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
                "generator_configurations": { "weighted":[False], "directed":[True,False],'bipartite':[True]},
                "encoder": unweighted_graph_encoder,
            },
            "kruskal": {
                "class": KruskalsAlgorithm,
                "input_type": "graph",
                "generator_configurations": { "weighted": [True], "directed":[False],'bipartite':[False] },
                "encoder": weighted_graph_encoder,
            }
        },
         "Randomised":{
             "karger":{
                 "class": KargerAlgorithm,
                 "input_type": "graph",
                 "generator_configurations": { "weighted": [False], "directed":[False],'bipartite':[False] },
                 "encoder": unweighted_graph_encoder,
             },
             "randomised_mst":{
                 "class": RandomisedMST,
                 "input_type": "graph",
                 "generator_configurations": { "weighted": [True], "directed":[False],'bipartite':[False] },
                 "encoder": weighted_graph_encoder,
             }
         },
         "Online":{
             "online_bipartite":{
                 "class": KruskalsAlgorithm,
                 "input_type": "graph",
                 "generator_configurations": { "weighted": [True], "directed":[False],'bipartite':[True] },
                 "encoder": weighted_graph_encoder,
             }         }
 }

node_sizes={"low_node_range":np.arange(3,11).tolist(),"mid_node_range":np.arange(12,21).tolist(),"high_node_range":np.arange(22,30).tolist()}

is_weighted={True:'weighted',False:'unweighted'}
is_directed={True:'directed',False:'undirected'}
