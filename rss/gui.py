import itertools as it
from math import pi as PI

# typing
from typing import override

import numpy as np
import pygame
import argparse
from numpy.typing import NDArray
from swarmsim.agent.MazeAgent import MazeAgent
from swarmsim.util.collider.AABB import AABB
from swarmsim.gui.agentGUI import DifferentialDriveGUI

from rss.graphing import extract_history, hr, label_vwx, plot_single_artists

matplotlib = None


def forward_axvspans(x, sense, offset=0):
    # create green vertical spanning regions for sensors
    #     [////]    [///]
    # xsen^    ^xnot
    xsen = [x[0]] if sense and sense[0] else []
    xnot = []
    for (xi, si), (xn, sn) in it.pairwise(zip(x, sense)):
        if sn > si:
            xsen.append(xn + offset)
        if si > sn:
            xnot.append(xi + offset)
    if sense and sense[-1]:
        xnot.append(x[-1])

    return xsen, xnot


class TennlabGUI(DifferentialDriveGUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fig = None
        self.axs = []
        self.artists = {}
        self.last_drawn = -1

    @override
    def draw(self, screen, draw_world=True):
        if draw_world:
            self.world.draw(screen)

        super().draw(screen)
        if pygame.font:
            if self.selected:
                a: MazeAgent = self.selected[0]
                if a.controller.neuron_counts is not None:
                    self.appendTextToGUI(screen, f"outs: {a.controller.neuron_counts}")

                if self.time > self.last_drawn:
                    self.plot_single(a)

                # if not was_plotted and self.has_plotted:
                #     self.update_legend()

                self.plt_update_show()

    def plt_update_show(self):
        if self.fig:
            self.fig.canvas.draw_idle()  # draw when events are processed
            self.fig.canvas.flush_events()  # process events
            # plt.pause(0.0001)

    def setup_graph_single(self, agent):
        if not self.check_matplotlib():
            return
        self.setup_figure()
        if not getattr(agent, 'history', False):
            return False

        cr = hr(0.0, 0.9, 0.4)
        cb = hr(0.6, 0.9, 0.4)
        cg = hr(0.3, 0.9, 0.4)
        ax, axw = self.axs

        x, _px, _py, _theta, sense, v, w = extract_history(agent)

        xsen, xnot = forward_axvspans(x, sense)

        # breakpoint()
        ar = self.artists
        ax.cla()
        axw.cla()
        ar['lineplot_v'] = ax.plot(x, v, c=cb, label="Velocity v", alpha=0.5)
        ar['lineplot_w'] = axw.plot(x, w, c=cr, label="Turn Rate $\\omega$", alpha=0.5)
        ar['lineplot_sense'] = ax.plot(x, sense, c=cg, label="Detection", alpha=0.1)
        # if plot_state:
        #     ax.subplot(111, aspect='equal')
        ar['axvspans_sense'] = [
            self.new_axvspan(ax, xa, xb)
            for xa, xb in zip(xsen, xnot)
        ]

        # for li in self.artists.values():
        #     for a in li:
        #         a.set_animated(True)

        self.update_legend()
        self.plt_update_show()

    def on_set_selected_single(self, agent):
        super().on_set_selected_single(agent)
        self.setup_graph_single(agent)

    def on_selected_event(self, prev, new):
        super().on_selected_event(prev, new)

    def plot_single(self, agent):
        ar = self.artists
        if not getattr(agent, 'history', False):
            return False
        firstplot = not self.artists
        if firstplot:
            self.setup_graph_single(agent)
        x, _px, _py, _theta, sense, v, w = extract_history(agent)
        ar['lineplot_v'][0].set_data(x, v)
        ar['lineplot_w'][0].set_data(x, w)
        ar['lineplot_sense'][0].set_data(x, sense)
        self.update_axvspans(self.axs[0], x, sense)
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view()
        # for li in self.artists.values():
        #     for a in li:
        #         a.axes.draw_artist(a)
        # if firstplot:
        # self.update_legend()
        self.last_drawn = self.time

    def new_axvspan(self, ax, xa, xb):
        return ax.axvspan(xa, xb, ymin=0.0, ymax=1.0, alpha=0.15, color='green')

    def update_axvspans(self, ax, x, sense):
        """Update the green vertical spans for the binary sensor state."""
        try:
            last = self.artists['axvspans_sense'][-1]  # the last green span
            x0 = last.get_x()  # the beginning of the last green span
        except IndexError:  # no green spans
            last = None
            x0 = 0  # check the whole history if no spans
        # Build the start/end pairs of green spans since the last update
        # including the last span that was updated.
        xsen, xnot = forward_axvspans(x[x0:], sense[x0:])
        # old: [////]    [///]
        # new: [////]    [///|/]   (/////) (///) <- new spans to be made
        #              x0^   |-> new
        #                |-->**| <- expand width of last
        for xs, xn in zip(xsen, xnot):
            if xs == x0 and last:
                last.set_width(xn - xs)  # expand the last green span
            else:  # make new green spans since last green span update
                self.artists['axvspans_sense'].append(
                    self.new_axvspan(ax, xs, xn)
                )

    @staticmethod
    def check_matplotlib():
        global matplotlib, plt
        if matplotlib is None:
            try:
                import matplotlib
                import matplotlib.pyplot as plt
            except ImportError:
                matplotlib = False
            plt.ion()
        return matplotlib

    def setup_figure(self):
        if not self.fig:
            self.fig, ax = plt.subplots()
            axw = ax.twinx()
            self.axs: list[plt.Axes] = [ax, axw]
            self.artists = {}
            self.fig.show()
        return self.fig, self.axs

    def update_legend(self):
        label_vwx((self.fig, self.axs),
                  title=f"Agent {self.selected[0].name} Sensor State and Speeds")


class VizTrailTennGUI(DifferentialDriveGUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trails = VizTrail(window=None)

    def draw(self, screen, draw_world=True):
        self.trails.draw(screen, self.world)
        if draw_world:
            self.world.draw(screen)

        super().draw(screen, False)


class VizTrail:
    def __init__(self, interval: int = 2, window: int | None = None, opacity=0.3, radius=1.0,
                 padding=1, limit=(0.1, 1000.0)):
        assert 0. <= opacity <= 1., "Opacity should be in [0, 1]"

        self.interval = interval
        self.window = window
        self.opacity = opacity
        self.agent_surfaces = []
        self.radius = radius
        self.padding = padding  # world coords
        self.limit = limit
        self.old_size = None

    # def population_aabb(population):
    #     return AABB([a.position for a in population])

    # def history_aabb(vectors):
    #     vectors = np.asarray(vectors)
    #     return AABB(vectors[:, :2])

    def zoom_fit_to_screen(self, screen, bb: AABB):
        # Compute bounding box and center
        bb = AABB(bb)
        tl = bb._min - self.padding
        br = bb._max + self.padding
        center = (tl + br) / 2.0
        tight_size = np.maximum(br - tl, 1e-5)  # Avoid div-by-zero

        screen_size = np.asarray(screen.get_size(), dtype=float)

        # determine zoom by choosing the widest dimension of the fitted box
        ideal_zoom = (screen_size / tight_size).min()
        new_zoom = np.clip(ideal_zoom, *self.limit)

        # 2. Correct Pan: align world center with screen center
        screen_center = screen_size / 2
        new_pan = screen_center - (center * new_zoom)
        return new_pan, new_zoom

    def vectorize_SE2_history(self, agent):
        history = agent.history[-self.window:self.interval:] if self.window else agent.history[:-self.interval:]
        return np.asarray([a[0] for a in history])

    def population_vectors(self, population):
        return np.asarray([self.vectorize_SE2_history(agent) for agent in population])

    @staticmethod
    def color_hsla(angle_rad: float, s=0.6, l=0.6, a=1.0, inplace: pygame.Color | None = None):
        color = pygame.Color('white') if inplace is None else inplace
        color.hsla = np.rad2deg(angle_rad) % 360., s * 100., l * 100., a * 100.
        return color

    def draw_surfaces(self, screen: pygame.Surface, vectors, offset=((0, 0), 1.0), world=None):
        pan, zoom = np.asarray(offset[0]), offset[1]
        size = screen.get_size()
        if self.old_size != size or len(self.surfaces) != len(vectors):
            self.surfaces = [pygame.Surface(size, pygame.SRCALPHA) for _ in vectors]
            self.old_size = size

        color = pygame.Color("white")
        if world is not None:
            radii = np.asarray([a.radius for a in world.population])
        else:
            radii = np.ones(len(vectors)) * self.radius
        for r, surf, history in zip(radii, self.surfaces, vectors):
            surf.fill("#00000000")
            for vec in history:
                if vec.size == 0:
                    continue
                pos, heading = vec[:2], vec[2]
                self.color_hsla(heading, a=self.opacity, inplace=color)
                pygame.draw.circle(surf, color, (pan + pos * zoom), r * zoom)

    def draw(self, screen: pygame.Surface, world, vectors=None, offset=None):
        if screen is None:
            return
        offset = (world.pos, world.zoom) if offset is None else offset
        pan, zoom = np.asarray(offset[0]), offset[1]

        screen.fill("#FFFFFFFF")

        if vectors is None:
            vectors = self.population_vectors(world.population)

        self.draw_surfaces(screen, vectors, offset=(pan, zoom), world=world)
        for surf in self.surfaces:  # layer surfaces on top of each other
            screen.blit(surf)
        for agent in world.population:  # draw opaque agents
            color = self.color_hsla(agent.angle, a=1.0)
            pygame.draw.circle(screen, color, (pan + agent.pos * zoom), agent.radius * zoom)


class EmptyAction(argparse.Action):
    def __init__(self, *args, empty_default=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.empty_default = empty_default

    def __call__(self, parser, namespace, values, option_string=None):
        if values in ('', None):
            values = self.empty_default
        setattr(namespace, self.dest, values)
