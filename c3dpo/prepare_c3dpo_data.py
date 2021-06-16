from glob import glob
import numpy as np
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True)
parser.add_argument("--name", help = "dataset name",required=True)
args = parser.parse_args()
sequences = glob(args.dataset)

data = {}
data['data'] = []

for name in sequences:
    folder = glob(name + '/*')
    for f in folder:
        #print(f)
        keypoints = glob(f + '/*.npy')
        for k in keypoints:
            #print("\t",k)
            kps = np.load(k)

            data_obj = {}
            data_obj['kp_loc'] = kps.T[:2,:].tolist()
            data_obj['kp_vis'] = kps.T[2,:].tolist()
            data_obj['K'] = [
                    [888.88,     0, 320],
                    [     0,  1000, 240],
                    [     0,     0,   1],
                    ]

            data['data'].append(data_obj)

data['dataset'] = args.name
print(json.dumps(data))
