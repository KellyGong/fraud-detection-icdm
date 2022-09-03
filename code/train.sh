# Parameters
model_id=1
gpu_id=5
icdm_sesison1_dir="dataset/"
icdm_sesison2_dir="dataset/"
ouput_result_dir="code/icdm_graph_competition/pyg_example/"
pyg_data_session1="dataset/pyg_data/icdm2022_session1"
pyg_data_session2="dataset/pyg_data/icdm2022_session2"
test_ids_session1="dataset/icdm2022_session1_test_ids.txt"
test_ids_session2="dataset/icdm2022_session2_test_ids.txt"

# Model hyperparameters
h_dim=256
n_bases=8
num_layers=3
fanout=150
n_epoch=5
early_stopping=6
lr=0.001
batch_size=64


python train.py --dataset $pyg_data_session1".pt" \
                --h-dim $h_dim \
                --n-bases $n_bases \
                --n-layers $num_layers \
                --fanout $fanout \
                --n-epoch $n_epoch \
                --early_stopping $early_stopping \
                --validation True \
                --lr $lr \
                --batch-size $batch_size \
                --model-id $model_id \
                --device-id $gpu_id
