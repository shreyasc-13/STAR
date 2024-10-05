from pathlib import Path
import numpy as np
import json
from copy import deepcopy

from ope.envs.asterix import AsterixEnv

from ope.experiment_tools.experiment import ExperimentRunner, aggregated_analysis, analysis, log_results
from ope.experiment_tools.config import Config
from ope.experiment_tools.factory import setup_params

from joblib import Parallel, delayed
import sys


def main(param):

    param = setup_params(param)

    n_arr = [int(5e1), int(1e2), int(5e2), int(1e3), int(5e3), int(1e4)]
    n_seeds = len(seeds)

    c_arr = [1, 2, 3, 4, 5]
    numz_arr = [2, 4, 8, 16, 32]

    for num_traj in n_arr:
        for c in c_arr:
            for numz in numz_arr:
                print(num_traj, c, numz)

                def _loop(num_traj, seed, c, numz):
                    runner = ExperimentRunner()

                    configuration = deepcopy(param['experiment'])
                    configuration['num_traj'] = num_traj
                    configuration['seed'] = seed  # np.random.randint(0, 2**16)
                    param['models']['ours']['k'] = c
                    param['models']['ours']['num_abs_states'] = numz

                    cfg = Config(configuration)
                    np.random.seed(cfg.seed)

                    env = AsterixEnv()

                    def processor(x): return x
                    absorbing_state = 2 * np.ones(400)

                    class policy:
                        def __init__(self, name):
                            self.name = name
                            if name == "uniform":
                                self.p = 1./env.n_actions * \
                                    np.ones(env.n_actions)
                            elif name == "skewed":
                                self.p = np.array(
                                    [0.01, 0.4, 0.05, 0.5, 0.02, 0.02])

                        def sample(self, states):
                            if self.name == "uniform":
                                return np.random.choice(env.n_actions, size=(len(states,)))
                            elif self.name == "skewed":
                                probs = self.p
                                return np.random.choice(env.n_actions, size=(len(states,)), p=probs)

                        def predict(self, states):
                            if self.name == "uniform":
                                return 1./env.n_actions * np.ones((len(states), env.n_actions))
                            elif self.name == "skewed":
                                probs = self.p * \
                                    np.ones((len(states), env.n_actions))
                                return probs

                        def get_action(self):
                            pass

                        def __call__(self, states):
                            return np.atleast_1d(self.sample(np.atleast_1d(states)))

                    pi_e = policy(name="skewed")
                    pi_b = policy(name="uniform")

                    cfg.add({
                        'env': env,
                        'pi_e': pi_e,
                        'pi_b': pi_b,
                        'processor': processor,
                        'absorbing_state': absorbing_state,
                    })
                    cfg.add({'models': param['models']})

                    runner.add(cfg)

                    results = runner.run()

                    log_file_name = Path(
                        f'heatmap_clustar_data/asterix/n_{num_traj}_c_{c}_numz_{numz}/seed_{seed}.json')
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
