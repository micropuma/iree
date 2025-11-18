iree-opt \
    --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-convert-dispatch-regions-to-workgroups))" \
    ./form_dispatch_workgroups.mlir \
    -o result.mlir
