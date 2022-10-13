# TRC-HCN
TRC-HCN: A Hypergraph Convolution Network Driven by the Trend of Relationship Change for Stock Ranking Prediction

## Data 
All datasets can be downloaded from: https://github.com/fulifeng/Temporal_Relational_Stock_Ranking/tree/master/data.
We can get the raw data through the links above. 

hyp_input_global.npy : global hypergraph of NASDAQ dataset.
hyp_input_clusters_T.zip : local hypergraph snapshots at different times of NASDAQ dataset.
priorKnowledge.npy : prior knowledge of NASDAQ dataset.

## Code
train_nasdaq.py: Train a model of TRC-HCN on NASDAQ dataset. 
The training procedure and evaluation procedure are all included in the `train_nasdaq.py`.
```train & evaluate
python train_nasdaq.py 
```
