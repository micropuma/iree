#!/bin/bash
iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-input-type=stablehlo \
             --mlir-print-ir-after-all \
             --mlir-pass-statistics \
             --mlir-timing \
             --iree-flow-dump-dispatch-graph-output-file=LSTM.dispatches.dot \
             LSTM.mlir -o LSTM.vmfb \
             2>&1 | tee output.dump

# dump dispatch graph for visualization
iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-input-type=stablehlo \
             --iree-flow-dump-dispatch-graph \
             --iree-flow-dump-dispatch-graph-output-file=LSTM.dispatches.dot \
             LSTM.mlir -o LSTM.vmfb

iree-run-module --device=cuda \
                --trace_execution \
                --module=LSTM.vmfb \
                --input=1x5xf32=[0,1,0,3,4] \
                --input=1x5x2x2xf32=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] \
                2>&1 | tee run-module.log
