#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  util.func public @no_batch_mmt4d_fusion(%arg0: tensor<1x1x64x1x1xf32>, %arg1: tensor<1x32x64x4x1xf32>, %arg2: tensor<1x1x32x1x4xf32>) -> tensor<1x1x32x1x4xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x32x1x4xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x32x1x4xf32>) -> tensor<1x1x32x1x4xf32>
    %2 = flow.dispatch.region -> (tensor<1x1x32x1x4xf32>) {
      %4 = linalg.batch_mmt4d ins(%arg0, %arg1 : tensor<1x1x64x1x1xf32>, tensor<1x32x64x4x1xf32>) outs(%1 : tensor<1x1x32x1x4xf32>) -> tensor<1x1x32x1x4xf32>
      flow.return %4 : tensor<1x1x32x1x4xf32>
    }
    %3 = flow.dispatch.region -> (tensor<1x1x32x1x4xf32>) {
      %4 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg2, %2 : tensor<1x1x32x1x4xf32>, tensor<1x1x32x1x4xf32>) outs(%0 : tensor<1x1x32x1x4xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %5 = arith.addf %in, %in_0 : f32
        linalg.yield %5 : f32
      } -> tensor<1x1x32x1x4xf32>
      flow.return %4 : tensor<1x1x32x1x4xf32>
    }
    util.return %3 : tensor<1x1x32x1x4xf32>
  }
}

