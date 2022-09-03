# Parameters
icdm_sesison1_dir="../data/session1/"
icdm_sesison2_dir="../data/session2/"
pyg_data_session1="../data/session1/icdm2022_session1"
pyg_data_session2="../data/session2/icdm2022_session2"
test_ids_session1="../data/session1/icdm2022_session1_test_ids.txt"
test_ids_session2="../data/session2/icdm2022_session2_test_ids.txt"

# sesison2 data generator
python format_pyg.py --graph=$icdm_sesison2_dir"icdm2022_session2_edges.csv" \
        --node=$icdm_sesison2_dir"icdm2022_session2_nodes.csv" \
        --storefile=$pyg_data_session2
