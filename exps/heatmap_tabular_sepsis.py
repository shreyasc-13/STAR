import pdb
import numpy as np
import argparse
import json
from copy import deepcopy

from ope.envs.sepsis import SepsisEnv

from ope.experiment_tools.experiment import ExperimentRunner, aggregated_analysis, analysis, log_results
from ope.experiment_tools.config import Config
from ope.experiment_tools.factory import get_model_from_name
from pathlib import Path

from joblib import Parallel, delayed
import sys


def main(param):

    # replace string of model with model itself in the configuration.
    for method, parameters in param['models'].items():
        if parameters['model'] != 'tabular':
            param['models'][method]['model'] = get_model_from_name(
                parameters['model'])

    n_arr = [int(5e1), int(1e2), int(5e2), int(1e3), int(5e3), int(1e4)]
    c_arr = [1, 2, 3, 4, 5]
    numz_arr = [2, 4, 8, 16, 32]
    n_seeds = len(seeds)

    for num_traj in n_arr:
        for c in c_arr:
            for numz in numz_arr:
                print(num_traj, c, numz)

                def _loop(num_traj, seed, c, numz):
                    runner = ExperimentRunner()

                    configuration = deepcopy(param['experiment'])
                    configuration['num_traj'] = num_traj
                    configuration['seed'] = seed
                    param['models']['ours']['k'] = c
                    param['models']['ours']['num_abs_states'] = numz

                    cfg = Config(configuration)

                    env = SepsisEnv(mdp_file=configuration['mdp_file'])

                    # set seed for the experiment
                    np.random.seed(cfg.seed)

                    # processor processes the state for storage
                    def processor(x): return x

                    # absorbing state for padding if episode ends before horizon is reached
                    absorbing_state = processor(np.array([env.n_dim - 1]))

                    # Setup policies
                    def _temp_scale(probs, tau):
                        log_probs = np.log(probs)
                        log_probs -= np.max(log_probs)
                        scaled_probs = np.exp(log_probs / tau)
                        scaled_probs /= np.sum(scaled_probs,
                                               axis=1, keepdims=True)
                        scaled_probs[np.isnan(scaled_probs)] = 0
                        return scaled_probs
                    data = np.load(configuration['mdp_file'])
                    expert_pi = data['expert_policy']

                    tau = 2
                    scaled_pi = _temp_scale(expert_pi, tau)

                    class policy:
                        def __init__(self, name, probs):
                            self.name = name
                            self.probs = probs

                        def sample(self, states):
                            if type(states) is np.ndarray:
                                probs = np.array(
                                    [self.probs[s.item()] for s in states])
                            elif type(states) is list:
                                probs = np.array([self.probs[s]
                                                 for s in states])
                            else:
                                raise "Policy initialization error."
                            return (probs.cumsum(1) > np.random.rand(probs.shape[0])[:, None]).argmax(1)

                        def predict(self, states):
                            if type(states) is np.ndarray:
                                probs = np.array(
                                    [self.probs[s.item()] for s in states])
                            elif type(states) is list:
                                probs = np.array([self.probs[s]
                                                 for s in states])
                            else:
                                raise "Policy initialization error."
                            return probs

                        def get_action(self):
                            pass

                        def __call__(self, states):
                            return np.atleast_1d(self.sample(np.atleast_1d(states)))

                    # Intialize behavior and evaluation policies with different temperature parameters
                    pi_e = policy(name="1_expert", probs=expert_pi)
                    pi_b = policy(name="2_expert", probs=scaled_pi)

                    cfg.add({
                        'env': env,
                        'pi_e': pi_e,
                        'pi_b': pi_b,
                        'processor': processor,
                        'absorbing_state': absorbing_state
                    })
                    cfg.add({'models': param['models']})

                    runner.add(cfg)

                    results = runner.run()

                    log_file_name = Path(
                        f'heatmap_clustar_data/sepsis/n_{num_traj}/n_{num_traj}_c_{c}_numz_{numz}/seed_{seed}.json')
                    log_file_name.parent.mkdir(parents=True, exist_ok=True)
                    log_results(log_file_name, results[0])

                Parallel(n_jobs=int(param['compute']['n_jobs']))(
                    delayed(_loop)(num_traj, seed, c, numz) for seed in seeds)


json_config = sys.argv[1]
json_seeds = sys.argv[2]
config = json.loads(json_config)
seeds = json.loads(json_seeds)

with open('cfgs/{0}'.format(config), 'r') as f:
    param = json.load(f)

print(seeds)

main(param)
