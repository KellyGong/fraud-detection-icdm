icdm_sesison1_dir="../data/session1/"
icdm_sesison2_dir="../data/session2/"
pyg_data_session1="../data/session1/icdm2022_session1"
pyg_data_session2="../data/session2/icdm2022_session2"
test_ids_session1="../data/session1/icdm2022_session1_test_ids.txt"
test_ids_session2="../data/session2/icdm2022_session2_test_ids.txt"


python inference.py --dataset  $pyg_data_session2".pt" \
               --test-file $test_ids_session2 \
               --model_id 70606