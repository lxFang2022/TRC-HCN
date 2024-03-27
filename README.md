# TD-HCN
TD-HCN: A Trend-Driven Hypergraph Convolutional Network for Stock Recommendation

## Data 
All datasets can be downloaded from: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master/data.

priorKnowledge.npy : prior knowledge of the NASDAQ dataset. Available through the above link.

hyp_input_global.npy : global hypergraph of the NASDAQ dataset. Available via PGRL.py.

hyp_input_clusters_T.zip : local hypergraphs at different times of the NASDAQ dataset. Available via PGRL.py.



## Code
train_nasdaq.py : Train a model of TD-HCN on the NASDAQ dataset. 
The training procedure and evaluation procedure are all included in the `train_nasdaq.py`.
```train & evaluate
python train_nasdaq.py 
```
