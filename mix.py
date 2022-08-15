from collections import defaultdict
import json
from statistics import mean


mixfiles = [
    'best_result/pyg_pred_session1_RGPRGNN_12270_0.930261.json',
    'best_result/pyg_pred_session1_RGPRGNN_14367_0.931277.json',
    'best_result/pyg_pred_session1_RGPRGNN_22588_0.9319.json',
    'best_result/pyg_pred_session1_RGPRGNN_64624_0.931947.json',
    'best_result/pyg_pred_session1_RGPRGNN_48223.json'
]

sample_id2values = defaultdict(list)

for mixfile in mixfiles:
    with open(mixfile, 'r') as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            
            y_dict = json.loads(line)
            sample_id2values[y_dict['item_id']].append(y_dict['score'])


with open('result.json', 'w+') as f:
    for sample_id in sample_id2values:
        y_dict = {}
        y_dict["item_id"] = int(sample_id)
        y_dict["score"] = float(mean(sample_id2values[sample_id]))
        json.dump(y_dict, f)
        f.write('\n')



