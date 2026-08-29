import getpass
import subprocess
import pandas as pd
import pathlib as pl
from itertools import product
from swarmsim.util.jinja import make_default_jinja_env


env = make_default_jinja_env(line_statement_prefix='#%:')

thisfile = pl.Path(__file__)
wd = thisfile.parent
template_path = wd / 'behavior.jinja.slurm'
scratch = pl.Path(f"/scratch/{getpass.getuser()}")


with open(template_path, 'r') as f:
    template = env.from_string(f.read())

eons_seeds = [2026, 2027, 2028]
swarm_sizes = [
    # 4,
    # 5,
    6,
    # 7,
    8,
    # 9,
    10,
    # 15,
    # 20,
    # 25,
    # 30,
    # 35,
    # 40,
    # 45,
    # 50,
]
rngstrats = [
    'TS1',
    # 'TSG',
    # 'TSR',
]

shortnames = {
    'Circliness': 'mill',
    'Aggregation': 'aggr',
    'ExplodingDispersion': 'disp',
    'DelaunayDiffusion': 'diff',
}
behaviors = [
    # 'Circliness',
    'Aggregation',
    'ExplodingDispersion',
    'DelaunayDiffusion',
]

configs = []
for behavior, swarm_size, eons_seed, rngstrat in product(
    behaviors, swarm_sizes, eons_seeds, rngstrats
):
    bhvr = shortnames[behavior]
    projname = f"{bhvr}-es{eons_seed}-{rngstrat}-n{swarm_size}"
    configs.append(dict(
        eons_seed=eons_seed,
        N=swarm_size,
        bhvr=bhvr,
        behavior=behavior,
        projname=projname,
        jobname=projname,
        projpath=scratch / f'{bhvr}/{swarm_size}' / projname,
        rngstrat=rngstrat,
    ))

print(pd.DataFrame(configs))
input('Press enter to continue or ctrl-c to cancel.')

for d in configs:
    slurm = template.render(**d)
    projpath = d['projpath']
    print(projpath)
    projpath.mkdir(parents=True, exist_ok=True)
    with open(f"{d['projpath']}/sbatch.slurm", 'w') as f:
        f.write(slurm)
    subprocess.run(f"sbatch {d['projpath']}/sbatch.slurm", shell=True)
