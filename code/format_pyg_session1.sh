# Parameters
icdm_sesison1_dir="../data/session1/"
icdm_sesison2_dir="../data/session2/"
pyg_data_session1="../data/session1/icdm2022_session1"
pyg_data_session2="../data/session2/icdm2022_session2"
test_ids_session1="../data/session1/icdm2022_session1_test_ids.txt"
test_ids_session2="../data/session2/icdm2022_session2_test_ids.txt"

# sesison1 data generator
python format_pyg.py --graph=$icdm_sesison1_dir"icdm2022_session1_edges.csv" \
        --node=$icdm_sesison1_dir"icdm2022_session1_nodes.csv" \
        --label=$icdm_sesison1_dir"icdm2022_session1_train_labels.csv" \
        --storefile=$pyg_data_session1
