import os
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cfg', help='Model config file path', dest='cfg_path')
args = vars(parser.parse_args())

config_path = args['cfg_path']
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")

print('======= TRAINING =======')
print(f'Using config: {config_path}')

os.system(f"python src/training.py --cfg {config_path}")