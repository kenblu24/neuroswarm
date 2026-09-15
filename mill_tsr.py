"""Generate data to plot how well an SNN performs across swarm sizes it wasn't trained on."""
import os
import re
import sys
import copy
import glob
import pathlib as pl
from functools import partial
from itertools import product

import tqdm
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

import common.experiment
import experiment_tenn2 as t2
from common.util import parse_rangelist
from common.project import UnzippedProject


wd = pl.Path(__file__).parent
cls = t2.ConnorMillingExperiment

desktop = pl.Path('/mnt/c/Users/kenbl/Desktop').expanduser()

folder = desktop / 'four' / 'mill'
globs = [
    # "/mnt/c/Users/kenbl/Desktop/four/mill/*/*",
    folder / "6/*",
    folder / "8/*",
    folder / "10/*",
    # "/scratch/kzhu4/mill/*/*",
    # "/scratch/kzhu4/aggr/*/*",
    # "/scratch/kzhu4/disp/*/*",
    # "/scratch/kzhu4/diff/*/*",
]
# include = r'mill.*TS1'
include = None
exclude = r'zip|\.7z'

projects = [UnzippedProject(p) for g in globs for p in glob.glob(str(g))
            if (exclude is None or not re.search(exclude, p)
                and (include is None or re.search(include, p)))]
# print(*[p.root for p in projects], sep='\n')


def get_parsers(parser, subpar):
    parser, subpar = t2.get_parsers(parser, subpar)
    sp = subpar.parsers

    sp['test'].add_argument('--rng_seed', type=int, default=None,
                                help="rng seed for the app")
    # sp['test'].add_argument('--Nrange', type=str, default=range(3, 50),
    #                             help="range of swarm sizes to test")
    sp['test'].add_argument('--trials', type=int, default=100,  # changed default from single
                                help="number of trials to run. Set to None to run one trial with world.yaml[seed]."
                                " Values greater than 0 will use the world.yaml[seed] to generate more seeds.")
    sp['test'].add_argument('--force', help="Skip confirmation prompts.", action='store_true')
    sp['test'].add_argument('--lastbest', help="test the last generation instead of the best.json", action='store_true')
    return parser, subpar


def single_fitness(args, seed):
    # seed = self.fetch_world_config().seed if seed is None else seed
    app = cls(args)
    world_final_state = app.simulate(None, app.net, seed=seed)
    # assert world_final_state.config.spawners[0]['n'] == n
    assert app.agents is not None
    assert world_final_state.seed is not None
    metric = app.pick_metric(world_final_state, app.args.behavior)
    rngstrat = app.p.experiment['args']['rngstrat']
    try:
        eons_seed = app.p.evolver['eons_params']['seed_eo']
    except KeyError:
        eons_seed = app.p.experiment['args']['eons_seed']
    return {
        'train_n': app.agents,
        'eons_seed': eons_seed,
        'seed': world_final_state.seed,
        'metric': metric.name,
        'rngstrat': rngstrat,
        'fitness': app.extract_fitness(world_final_state, metric),
    }


def mp_fitness(bundle):
    app, seed = bundle
    return single_fitness(app, seed=seed)


def test(args, silent=False):
    def prnt(*args, **kwargs):
        if not silent:
            print(*args, **kwargs)

    # Set up simulator and network
    # proc = None
    # net = None if args.stdin == 'stdin' else app.net

    # projects = [UnzippedProject(p)
    #             for n in folder.iterdir() if n.is_dir() and 'zip' not in n.name
    #             for p in n.iterdir()]

    args_copies = []
    for project in projects:
        args_copy = copy.deepcopy(args)
        assert project.possibly_valid()
        args_copy.project = str(project.root)
        args_copy.root = None
        # if args.lastbest:
        last_networks = list(project.networks[500].values())
        assert len(last_networks) == 1
        args_copy.network = last_networks[0]
        args_copies.append(args_copy)

    if args.trials is not None:
        # if the yaml has null seed, or if --rngstrat TSR
        seeds = np.random.default_rng(args.rng_seed).integers(0, 2**32, size=args.trials)
    else:
        seeds = [args.rng_seed]
    prnt(seeds)
    bundles = tuple(product(args_copies, seeds))
    pd.options.display.max_colwidth = 128
    pd.options.display.max_rows = 200
    pd.options.display.min_rows = 200
    prnt(pd.DataFrame(bundles))
    if not args.force:
        input("Press enter to continue, ctrl-c to cancel.")

    if args.processes == 1 or (args.processes is None and os.cpu_count() == 1):
        prnt(f"Using single thread.")
        results = [single_fitness(*bundle) for bundle in tqdm.tqdm(bundles)]
    else:
        if args.processes is None:
            prnt(f"Using {os.cpu_count()} detected CPUs/threads.")
        else:
            prnt(f"Using {args.processes} threads.")

        # app handles making seeds based on number of trials from args
        results = process_map(mp_fitness, bundles, max_workers=args.processes)

        # for res in results:
        #     prnt(f"agents trained with {res['train_n']:2d}\tSeed {res['seed']}"
        #          f"\tFitness ({res['metric']}): {res['fitness']:8.4f}")

        df = pd.DataFrame(results)
        df.to_csv(wd / "results/test.csv")
        # print(f"Sum: {sum(fitness):8.4f} \t Avg: {sum(fitness) / len(fitness):8.4f} \t Std: {np.std(fitness):8.4f}")
        # print(f"Min: {min(fitness):8.4f} \t Max: {max(fitness):8.4f} \t out of {len(fitness)} trials")

    return df


if __name__ == "__main__":
    parser, subpar = get_parsers(*t2.get_parsers(*common.experiment.get_parsers()))
    thisfile, *argv = sys.argv
    args = parser.parse_args(['test', *argv])
    args.environment = "mill-n-vs-seed-v01"
    test(args)
