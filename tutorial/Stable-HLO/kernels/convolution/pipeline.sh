iree-compile --iree-hal-target-backends=cuda \
             --iree-cuda-target=sm_86 \
             --iree-input-type=stablehlo \
             --mlir-print-ir-after-all \
             convolution.mlir -o convolution.vmfb \
             2>&1 | tee output.dump