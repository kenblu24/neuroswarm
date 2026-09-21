import sys
import itertools as it

import matplotlib.pyplot as plt

# from rss.gui import TennlabGUI
import experiment_tenn2 as t2
from common import experiment
import rss.graphing as graphing

# typing:
from typing import override
from swarmsim.world.RectangularWorld import RectangularWorld


def graph_each(world, start_offset=0, end_offset=None, length=None):
    bundles = []
    for i, agent in enumerate(world.population):
        fig, ax = plt.subplots()
        axw = ax.twinx()
        bundle = graphing.plot_single_artists(agent, fig, (ax, axw),
                    start_offset=start_offset, end_offset=end_offset, length=length)
        bundles.append(bundle)
        _h, _l, legend = graphing.label_vwx(
            (fig, (ax, axw)),
            title=f"Agent {i} Sensors vs. Control Inputs",
            columnspacing=0.75,
        )
        bundle[2]['lineplot_sense'][0].set_visible(False)
        ax.relim(visible_only=True)
        ax.margins(y=0.1)
        axw.margins(y=0.2)
        # make it smaller
        fig.set_figwidth(4.5)
        fig.set_figheight(2)
        fig.subplots_adjust(top=0.70, bottom=0.22)
        fig.subplots_adjust(left=0.17, right=0.80)
        legend.set_bbox_to_anchor((0.5, 0.90))
    return bundles


def run(app: t2.ConnorMillingExperiment, args):

    # Set up simulator and network
    import pygame
    from PIL import Image
    import numpy as np

    proc = None
    net = app.net

    i = 0

    from swarmsim.world.subscribers.WorldSubscriber import WorldSubscriber

    def save_frames(world, screen):
        nonlocal i

        if i % 200 != 0 and i != 1:
            i += 1
            return

        frame = pygame.surfarray.array3d(screen)
        img = Image.fromarray(np.uint8(frame))
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = img.rotate(90, expand=True)
        img = img.crop((0, 0, img.height, img.height))

        img.save(app.p.ensure_file_parents(f"frames/frame_{i}.png"))
        i += 1

    def modify_simargs(app, simargs):
        world_subscribers = simargs['subscribers']
        simargs['world_config'].background_color = (255, 255, 255)
        world_subscribers.append(WorldSubscriber(func=save_frames))
        return simargs

    # Run app and print fitness
    world = app.simulate(proc, net, init_callback=modify_simargs)
    fitness = app.extract_fitness(world)
    print(f"Fitness: {fitness:8.4f}")

    import matplotlib.pyplot as plt
    # fig = graphing.plot_multiple(world)
    # fig.suptitle('')
    kwargs = dict(start_offset=args.offset, end_offset=args.offset_end, length=args.length)
    try:
        for fig, _axs, _artists in graph_each(world, **kwargs):
            fig.savefig(app.p.ensure_file_parents(f"plots/agent_trajectories_{fig.number}.pdf"))
        graphing.export(world, output_file=app.p.ensure_file_parents("agent_trajectories.xlsx"))
    except ValueError:
        plt.close()

    fig, _ax = graphing.plot_fitness(world)
    # fig.suptitle('')
    # make it smaller
    fig.set_figheight(2)
    fig.subplots_adjust(bottom=0.22)
    fig.subplots_adjust(left=0.11, right=0.95)
    fig.savefig(app.p.ensure_file_parents(f"plots/fitness.pdf"))

    fig, ax = plt.subplots()
    cr = graphing.hr(0.0, 0.9, 0.4)
    cb = graphing.hr(0.6, 0.9, 0.4)
    cg = graphing.hr(0.3, 0.9, 0.4)
    agent = world.population[3]

    x, _px, _py, _theta, sense, v, w = graphing.extract_history(agent)

    # create green vertical spanning regions for sensors
    xsen = [1] if sense and sense[0] else []
    xnot = []
    for (xi, si), (xn, sn) in it.pairwise(zip(x, sense)):
        if sn > si:
            xsen.append(xn)
        if si > sn:
            xnot.append(xi)
    if sense and sense[-1]:
        xnot.append(len(sense) - 1)
    ax.plot(x, sense, c=cg, label="Detection", alpha=0.1)
    legend = fig.legend(loc='upper center', ncol=3, fancybox=True, shadow=True,
                        bbox_to_anchor=(0.5, 0.90))
    # legend.set_bbox_to_anchor((0.5, 0.95))
    fig.subplots_adjust(top=0.87, hspace=0.08, right=(1 - fig.subplotpars.left), bottom=0.1)
    for xa, xb in zip(xsen, xnot):
        ax.axvspan(xa, xb, ymin=0.0, ymax=1.0, alpha=0.15, color='green')
    fig.suptitle(f"Agent {agent.name} Sensor State")
    # make it smaller
    fig.set_figheight(2)
    fig.subplots_adjust(top=0.70, bottom=0.22)
    fig.subplots_adjust(left=0.11, right=0.88)
    s = legend.legend_handles[-1]  # the last column should be the vertical lines.
    s.set_linewidth(10.0)  # Draw that thicker in the legend   # pyright: ignore[reportAttributeAccessIssue]
    s.set_alpha(0.25)
    fig.savefig(app.p.ensure_file_parents(f"plots/agent_sensors_{agent.name}.pdf"),
                # backend='pgf',
    )

    if args.explore:
        app.p.explore()
    plt.show(block=True)
    # TODO: handle when no project

    return fitness


def get_parsers(parser, subpar):
    sp = subpar.parsers['run']
    sp.add_argument("--offset", type=int, help="Number of steps at the start of the file to skip", default=0)
    sp.add_argument("--offset_end", type=int, help="Number of steps at the end of the file to ignore",)
    sp.add_argument("--length", type=int, help="Length of time to graph in steps",)
    return parser, subpar


if __name__ == "__main__":
    parser, subpar = get_parsers(*t2.get_parsers(*experiment.get_parsers()))
    thisfile, *argv = sys.argv
    args = parser.parse_args(['run', *argv])
    args.environment = 'plot-tenn2'
    app = t2.ConnorMillingExperiment(args)
    run(app, args)
