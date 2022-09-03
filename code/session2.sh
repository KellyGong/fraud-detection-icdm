icdm_sesison1_dir="../data/session1/"
icdm_sesison2_dir="../data/session2/"
pyg_data_session1="../data/session1/icdm2022_session1"
pyg_data_session2="../data/session2/icdm2022_session2"
test_ids_session1="../data/session1/icdm2022_session1_test_ids.txt"
test_ids_session2="../data/session2/icdm2022_session2_test_ids.txt"


python main.py --dataset  $pyg_data_session1".pt" \
               --alpha 0.5 \
               --batch_size 256 \
               --cl \
               --cl_batch 2048 \
               --cl_common_lr 0.002 \
               --cl_epoch 3 \
               --cl_finetune_lr 0.005 \
               --cl_joint_loss True \
               --cl_lr 0.002 \
               --h_dim 64 \
               --lr 0.002 \
               --model_id 2 \

