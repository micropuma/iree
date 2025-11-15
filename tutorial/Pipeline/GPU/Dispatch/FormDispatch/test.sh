iree-opt \
    --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" \
    --split-input-file \
    ./form_dispatch_regions.mlir \
    -o form_dispatch_regions_result.mlir

iree-opt \
    --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" \
    --split-input-file \
    ./mmt4d_fusion.mlir \
    -o mmt4d_fusion_result.mlir

iree-opt \
    --pass-pipeline="builtin.module(util.func(iree-dispatch-creation-form-dispatch-regions{aggressive-fusion=true}))" \
    --split-input-file \
    ./consumer_fusion.mlir \
    -o consumer_fusion_result.mlir