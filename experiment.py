import subprocess
import argparse
import os
import json


"""Parse arguments for the experiment script."""

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--list', nargs='*', type=str)
parser.add_argument('-d', '--domain', nargs='*', type=str)
args = parser.parse_args()
seeds = args.list
domain = args.domain[0]


def _trim(x):
    if ',' in x:
        return int(x[:-1])
    else:
        return int(x)


seeds = [_trim(s) for s in seeds]

print(domain)

"""Run script corresponding to provided experiment name."""

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['PYTHONPATH'] = current_dir

if domain == "cartpole":
    cfg_file = "nn_cartpole_cfg.json"
    subprocess.run(["python", "exps/nn_cartpole.py",
                   json.dumps(cfg_file), json.dumps(seeds)])
elif domain == "cartpole_heatmap":
    cfg_file = "nn_cartpole_cfg.json"
    subprocess.run(["python", "exps/heatmap_nn_cartpole.py",
                   json.dumps(cfg_file), json.dumps(seeds)])
elif domain == "sepsis":
    cfg_file = "tabular_sepsis_cfg.json"
    subprocess.run(["python", "exps/tabular_sepsis.py",
                   json.dumps(cfg_file), json.dumps(seeds)])
elif domain == "sepsis_heatmap":
    cfg_file = "tabular_sepsis_cfg.json"
    subprocess.run(["python", "exps/heatmap_tabular_sepsis.py",
                   json.dumps(cfg_file), json.dumps(seeds)])
elif domain == "asterix":
    cfg_file = "nn_asterix_cfg.json"
    subprocess.run(["python", "exps/nn_asterix.py",
                   json.dumps(cfg_file), json.dumps(seeds)])
elif domain == "asterix_heatmap":
    cfg_file = "nn_asterix_cfg.json"
    subprocess.run(["python", "exps/heatmap_nn_asterix.py",
                   json.dumps(cfg_file), json.dumps(seeds)])
