"""Generate color trails images in project(s) after training.

Takes a path string of project folders as positional arguments.

If you're processing multiple projects, use --noviz to avoid spawning
windows for visualization.

If you really want to see a single window, pass -p 1 to disable multiprocessing.

Control the size of the image with --viz_trails <width>x<height>. Defaults to 2000x2000.

Exclude paths with --exclude <regex>.

All other arguments are passed to the experiment_tenn2.py argument parser.

Examples:
    python bulkgen_colortrails.py '/mnt/c/Users/kenbl/Desktop/aggr/*/*' --noviz --exclude zip
    python bulkgen_colortrails.py config/241104-121312-connorsim_snn_eons-v01 -p 1
    python bulkgen_colortrails.py results/mill/* --noviz --viz_trails 400x400 --cycles 2000 --caspian

Note: If using SLURM `srun`, you should use unbuffered mode:
    srun -c 64 -t 00:30:00 python -u bulkgen_colortrails.py ...
or set PYTHONUNBUFFERED=1 in your environment.

"""
import os
import re
import sys
import copy
import glob
from warnings import warn

import tqdm
from tqdm.contrib.concurrent import process_map

import common.experiment
import experiment_tenn2 as t2
from common.project import UnzippedProject


cls = t2.ConnorMillingExperiment


def get_parsers(parser, subpar):
    parser, subpar = t2.get_parsers(parser, subpar)
    sp = subpar.parsers
    sp['run'].add_argument('project', nargs='+',
                           help="Specify globs to projects to generate images for.")
    sp['run'].add_argument('-p', '--processes', type=int, default=None,
                           help="number of threads for concurrent fitness evaluation. Defaults to detected CPU count.")
    sp['run'].add_argument('--viz_trails', default='2000x2000', help="Specify a size for the screenshot, e.g. 800x800.")
    sp['run'].add_argument('--exclude', help="regex to exclude paths. Applied per discovered path")
    sp['run'].add_argument('-y', '--force', help="Skip confirmation prompts.", action='store_true')
    return parser, subpar


def run_one(args):
    # seed = self.fetch_world_config().seed if seed is None else seed
    app = cls(args)
    t2.run(app, args, silent=True)


def main(args, silent=False):
    def prnt(*args, **kwargs):
        if not silent:
            print(*args, **kwargs)

    if args.root != common.experiment.DEFAULT_PROJECT_BASEPATH:
        raise ValueError("Can't use --root with this script. Put a glob of your folders as the first argument.")
    if not args.noviz and (args.processes is None or args.processes > 1):
        raise ValueError("Can't use multiprocessing with visualization. Use --noviz, or set -p 1")

    projects = [UnzippedProject(p)
                for globstr in args.project for p in glob.glob(globstr)
                if args.exclude is None or not re.search(args.exclude, p)]

    args_copies = []
    skipped = 0
    for project in projects:
        if not project.possibly_valid():
            skipped += 1
            msg = f"Invalid project.\n\t{project.root} \tis not a valid project. Skipping."
            warn(msg, stacklevel=1)
            continue
        args_copy = copy.deepcopy(args)
        args_copy.project = str(project.root)
        args_copy.root = None
        args_copies.append(args_copy)

    prnt("Will generate images for the following projects:\n")
    prnt(*[a.project for a in args_copies], sep='\n')
    if skipped:
        prnt(f"WARNING: Skipped {skipped} matches.")
    if not args.force:
        prnt("Press enter to continue, ctrl-c to cancel.")
        input()  # empty input() for `srun`
    prnt()

    if args.processes == 1 or (args.processes is None and os.cpu_count() == 1):
        prnt(f"Using single thread.")
        for arg in tqdm.tqdm(args_copies):
            run_one(arg)
    else:
        if args.processes is None:
            prnt(f"Using {os.cpu_count()} detected CPUs/threads.")
        else:
            prnt(f"Using {args.processes} threads.")
        process_map(run_one, args_copies, max_workers=args.processes)


if __name__ == "__main__":
    parser, subpar = get_parsers(*t2.get_parsers(*common.experiment.get_parsers()))
    thisfile, *argv = sys.argv
    args = parser.parse_args(['run', *argv])
    args.environment = "bulkgen_colortrails-v01"
    main(args)
