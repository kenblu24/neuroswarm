import numpy as np
from .CasPyanBinaryController import CasPyanBinaryController

# typing
from typing import Any, override


class CasPyanBinaryRemappedController(CasPyanBinaryController):
    def __init__(self, *args, resolve_ties=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.resolve_ties = resolve_ties

    @override
    def run_processor(self, observation):
        b2oh = self.bool_to_one_hot

        # translate observation to vector
        input_vector = b2oh(observation)
        # input_vector += (1,)  # add 1 as constant on input to 4th input neuron

        # encode to spikes
        input_slice = input_vector[:len(self.encoder)]
        spikes = [enc.get_spikes(x) for enc, x in zip(self.encoder, input_slice)]
        # run processor
        self.processor.apply_spikes(spikes)
        self.processor.run(self.extra_ticks)
        if self.neuro_track_all:
            neuron_counts = np.asarray(self.processor.neuron_counts())
        self.processor.run(self.neuro_tpc)
        if self.neuro_track_all:
            neuron_counts += self.processor.neuron_counts()
            self.neuron_counts = neuron_counts.tolist()
        # action: bool = bool(proc.output_vectors())  # old. don't use.
        data = [dec.decode(node.history) for dec, node in zip(self.decoder, self.processor.outputs)]
        data = [int(round(x)) for x in data]
        # three bins. One for +v, -v, omega.
        # v = self.scale_v * (data[1] - data[0])
        # w = self.scale_w * (data[3] - data[2])
        # these values were taken from an average of speeds/turning rates
        # from measurements of Turbopis 1, 2, 3, 4 @ (100, 90, +-0.5)
        v_mapping = [0.0, 0.276,]
        w_mapping = [0.0, 0.602,]
        v = v_mapping[data[1]] - v_mapping[data[0]]
        w = w_mapping[data[3]] - w_mapping[data[2]]

        if self.resolve_ties == 0:
            # Do nothing
            pass
        elif self.resolve_ties == 1:
            # Use RNG to split ties
            if v == 0.0:
                v = v_mapping[1] * self.parent.rng.choice([-1, 1])
            if w == 0.0:
                w = w_mapping[1] * self.parent.rng.choice([-1, 1])
        elif self.resolve_ties == 2:
            # Force v to -1, w to -1
            if v == 0.0:
                v = -v_mapping[1]
            if w == 0.0:
                w = -w_mapping[1]
        elif self.resolve_ties == 3:
            # Force v to -1, w to 1
            if v == 0.0:
                v = -v_mapping[1]
            if w == 0.0:
                w = w_mapping[1]
        elif self.resolve_ties == 4:
            # Force v to 1, w to -1
            if v == 0.0:
                v = v_mapping[1]
            if w == 0.0:
                w = -w_mapping[1]
        elif self.resolve_ties == 5:
            # Force v to 1, w to 1
            if v == 0.0:
                v = v_mapping[1]
            if w == 0.0:
                w = w_mapping[1]
        elif self.resolve_ties == 6:
            if v == 0.0 or w == 0.0:
                v, w = 0.0, 0.0
        else:
            raise ValueError("resolve_ties should be a choice from 0-6")

        return v, w
        # return (0.08, 0.4) if not observation else (0.18, 0.0)  # CMA best controller
