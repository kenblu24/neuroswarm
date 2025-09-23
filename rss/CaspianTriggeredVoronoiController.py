"""
SNN controller that triggers when agent should update its waypoint

Use with sensor_id pointing to a RelativeAgentSensor on the agent
And the world should have a VoronoiRelaxation metric in world.metrics
"""

import math
import numpy as np

# typing
from typing import Any, override, TYPE_CHECKING

from swarmsim.agent.MazeAgent import MazeAgent
from swarmsim.sensors.RelativeAgentSensor import RelativeAgentSensor
from .CaspianBinaryController import CaspianBinaryController
from swarmsim.util import statistics_tools as st
from swarmsim.util.pid import PID

if TYPE_CHECKING:
    from swarmsim.metrics import VoronoiRelaxation
else:
    VoronoiRelaxation = None

import neuro
import caspian


class CaspianTriggeredVoronoiController(CaspianBinaryController):

    def __init__(
        self,
        agent,
        parent=None,
        network: dict[str, Any] | None = None,
        neuro_tpc: int | None = None,
        extra_ticks: int | None = None,
        neuro_track_all: bool = False,
        scale_forward_speed: float = 0.2,  # m/s forward speed factor
        scale_turning_rates: float = 2.0,  # rad/s turning rate factor
        sensor_id: int = 0,
    ) -> None:
        # if config is None:
        #     config = MazeAgentCaspianConfig()

        super().__init__(
            agent=agent,
            parent=parent,
            network=network,
            neuro_tpc=neuro_tpc,
            extra_ticks=extra_ticks,
            neuro_track_all=neuro_track_all,
            scale_forward_speed=scale_forward_speed,
            scale_turning_rates=scale_turning_rates,
            sensor_id=sensor_id,
        )

        self.vpid = PID(0.3, 0.0, 0)
        self.wpid = PID(0.3, 0.0, 0)

        self.setpoint = None
        self.vor_metric = None
        self.nearest_centroid = None

    # @classmethod  # to get encoder structure/#neurons for external network generation (EONS)
    # def get_default_encoders(cls, neuro_tpc=None):
    #     encoder_neurons, decoder_neurons = cls.default_inputs, cls.default_outputs
    #     if neuro_tpc is None:
    #         neuro_tpc = cls.default_neuro_tpc
    #     encoder_params = {
    #         "dmin": [0] * encoder_neurons,  # two bins for each binary input + random
    #         "dmax": [1] * encoder_neurons,
    #         "interval": neuro_tpc,
    #         "named_encoders": {"s": "spikes"},
    #         "use_encoders": ["s"] * encoder_neurons
    #     }
    #     decoder_params = {
    #         # see notes near where decoder is used
    #         "dmin": [0] * decoder_neurons,
    #         "dmax": [1] * decoder_neurons,
    #         "divisor": neuro_tpc,
    #         "named_decoders": {"r": {"rate": {"discrete": False}}},
    #         "use_decoders": ["r"] * decoder_neurons
    #     }
    #     encoder = neuro.EncoderArray(encoder_params)
    #     decoder = neuro.DecoderArray(decoder_params)

    #     return (
    #         encoder.get_num_neurons(),
    #         decoder.get_num_neurons(),
    #         encoder,
    #         decoder
    #     )

    def run_processor(self, observation, scale_input: float = 1.0):
        positions = observation
        positions = np.asarray(sorted(list(positions), key=np.linalg.norm))
        flat = np.zeros(self.n_inputs)  # eventual input vector
        # flatten the positions. If there are too many to fit our input vector, just use the first n_inputs-1 positions
        data = positions.flatten()[:(self.n_inputs - 1)]
        data *= scale_input  # normalize the inputs so it fits encoder range
        flat[:len(data)] = data  # copy the normalized inputs into the input vector
        flat[-1] = 1.  # add a constant input to the last input for spikes on every processor tick
        spikes = self.encoder.get_spikes(flat)
        self.processor.apply_spikes(spikes)
        self.processor.run(self.extra_ticks)
        if self.neuro_track_all:
            neuron_counts = np.asarray(self.processor.neuron_counts())
        self.processor.run(self.neuro_tpc)
        if self.neuro_track_all:
            neuron_counts += self.processor.neuron_counts()
            self.neuron_counts = neuron_counts.tolist()
        # action: bool = bool(proc.output_vectors())  # old. don't use.
        data = self.decoder.get_data_from_processor(self.processor)
        return data[0] > 0.5

    def cache_world_voronoi(self, world):
        # find and cache the VoronoiRelaxation metric from the world.metrics list
        if self.vor_metric is not None:
            return self.vor_metric
        from swarmsim.metrics import VoronoiRelaxation
        for metric in world.metrics:
            if isinstance(metric, VoronoiRelaxation):
                self.vor_metric = metric
                return metric
        raise RuntimeError("Could not find VoronoiRelaxation metric in world.")

    def target_centroid(self, vor_metric: VoronoiRelaxation, agent):
        """Get the centroid corresponding to this agent"""
        # This is currently calculated by finding the closest centroid to the agent
        # This is not necessarily the centroid of the agent's current voronoi cell
        # THIS IS WHAT IS BROKEN
        # TODO: Actually get the centroid of the agent's current voronoi cell
        if not vor_metric.vor:
            return None  # If voronoi tesselation hasn't been calculated yet, there is no target centroid
        distances = np.array([np.linalg.norm(vert - agent.getPosition()) for vert in vor_metric.vor.filtered_points])
        if distances.size == 0:
            return None  # If there are no points in the voronoi tesselation, there is no target centroid
        idx = distances.argmin()
        return vor_metric.centroids[idx]

    def get_actions(self, agent: MazeAgent) -> tuple[float, float]:
        sensor: RelativeAgentSensor = self.parent.sensors[self.sensor_id]

        # Ask the processor if, given the current sensor reading, the agent should
        # query the voronoi tesselation for an updated target centroid location
        self.triggered = self.run_processor(sensor.current_state, 1 / sensor.r)
        self.triggered = 1
        agent.set_color_by_id(self.triggered)
        if self.triggered:
            vm: VoronoiRelaxation = self.cache_world_voronoi(agent.world)
            self.setpoint = self.target_centroid(vm, agent)

        if self.setpoint is None:
            return 0, 0  # no move

        # basic PID control to go towards setpoint
        ego = self.setpoint - agent.pos  # vector from target to agent
        angleto = np.arctan2(ego[1], ego[0])  # global heading of above vector
        derr = np.linalg.norm(ego)  # distance to target
        werr = (angleto % (2 * np.pi)) - (agent.angle % (2 * np.pi))  # angle to target
        # PID control
        v = self.vpid(derr)  # slow down when approaching target
        w = self.wpid(werr)  # turn towards target
        # limit the speeds
        v = np.clip(v, -self.scale_v, self.scale_v)
        w = np.clip(w, -self.scale_w, self.scale_w)

        self.requested = v, w
        return self.requested

    def draw(self, screen, offset):
        # RSS doesn't call draw() on controllers YET so this is a placeholder
        super().draw(screen, offset)
