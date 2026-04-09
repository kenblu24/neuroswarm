#!/bin/bash

cd ../..

# mktemp -d

python - train <<'EOF'
import experiment_tenn2 as t2
class HeredocExperiment(t2.ConnorMillingExperiment):
    @staticmethod
    def init_callback(self, simargs):
        simargs['config']['spawners'][0]['agent']

        return simargs
t2.main(cls=HeredocExperiment)
EOF