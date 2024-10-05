import numpy as np
import subprocess
import os

# Number of jobs to launch on cluster
srun_num_jobs = 25

# Number of seeds to average experiments over
seeds = np.arange(1, 501)

# Name of experiment
domain = "asterix_heatmap"
assert (domain in ["cartpole", "cartpole_heatmap", "sepsis",
        "sepsis_heatmap", "asterix", "asterix_heatmap"])

# Split seeds across jobs and launch all
for i, seed_split in enumerate(np.array_split(seeds, srun_num_jobs)):
    str_ss = str(list(seed_split))
    str_ss = str_ss[1:]
    str_ss = str_ss[:-1]
    cmd = f'screen -dmS {domain}_{i} bash -c "source ~/.bashrc; conda activate <conda_env>; srun \
    --partition <partition_name> \
        --mem=64G \
            --cpus-per-task=8 \
                --time=8-00:00:00 \
                    -u python experiment.py -l {str_ss} -d {domain}; \
                        exec sh"'
    process = subprocess.Popen(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
