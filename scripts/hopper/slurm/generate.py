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
    4,
    5,
    6,
    # 7,
    # 8,
    # 9,
    10,
    # 15,
    # 20,
    # 25,
    # 30,
    # 35,
    # 40,
    # 45,
    50,
]
bhvr = 'mill'
behavior = 'Circliness'

configs = []
for eons_seed, swarm_size in product(eons_seeds, swarm_sizes):
    projname = f"{bhvr}-es{eons_seed}-t1-n{swarm_size}"
    configs.append(dict(
        eons_seed=eons_seed,
        N=swarm_size,
        bhvr=bhvr,
        behavior=behavior,
        projname=projname,
        projpath=str(scratch / f'{bhvr}/{swarm_size}' / projname),
    ))

pd.DataFrame(configs)  # show

for d in configs:
    slurm = template.render(**d)
    print(slurm)
    projpath = d['projpath']
    print(projpath)
    projpath.mkdir(parents=True, exist_ok=True)
    with open(f"{d['projpath']}/sbatch.slurm", 'w') as f:
        f.write(slurm)
    subprocess.run(f"sbatch {d['projpath']}/sbatch.slurm", shell=True)
