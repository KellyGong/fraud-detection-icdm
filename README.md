# PyG example

### Include

+ `pyg_demo.sh`: Shell Script (include converting raw data into PyG graph, training model and generating final result)
+ `rgcn_mb_icdm.py`: PyG rgcn model
+ `format_pyg.py`: PyG data generator
+ `best_model/`: Directory to store model trained 





### How to run

+ Run shell script

```
mkdir -p /dataset/pyg_data/
cd /code/icdm_graph_competition/pyg_example/
sh pyg_demo.sh
```

+ Output

  1. PyG Graph is stored in `/dataset/pyg_data/`

     + `icdm2022_session1.pt` PyG Heterograph of session1 data

     + `icdm2022_session2.pt` PyG Heterograph of session2 data
     + `*.nodes.pyg ` Temporary cache file (Removable)

  2. Trained Model is saved in `/code/icdm_graph_competition/best_model/`

     + (Default) `1.pth` Model file
     + (Optional) change `model_id` to customize name of model file

  3. Final result of Inference is generated in `/code/icdm_graph_competition/pyg_example/`

     + `pyg_session1.json` Result of session1
     + `pyg_session2.json` Result of session2


# icdm 2022 baseline

We implement [RCCN](https://arxiv.org/abs/1703.06103) baseline on [ICDM 2022 : Risk Commodities Detection on Large-Scale E-commerce Graphs](https://tianchi.aliyun.com/competition/entrance/531976/introduction). Our implementation include three popular GNN platforms: [DGL](https://github.com/dmlc/dgl), [PyG](https://github.com/pyg-team/pytorch_geometric) and [OpenHGNN](https://github.com/BUPT-GAMMA/OpenHGNN). You can use it as a reference for data processing, model training and inference.

## Environment Setup

- **Option1:** Run on OpenI Cloud platform.
  - We provide an image with pytorch, dgl and pyg installed. You can directly create an instance from that. Also, you can use dataset provided on OpenI.
- **Option2:** Run locally.
  - Download the dataset and config the environment locally.

## Experiment Result

| AP       | session1 | session2 |
|----------|----------|----------|
| pyg      | 0.9174   | 0.8931   |               
| dgl      | 0.8751   | 0.8417   |               
| openhgnn | 0.8658   | 0.8387   |     