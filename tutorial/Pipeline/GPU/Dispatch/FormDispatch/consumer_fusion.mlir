util.func @custom_op_consumer_fusion(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %0 = iree_linalg_ext.custom_op {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]}
      ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : tensor<?x?xf32>, %b1 : tensor<?xf32>):
      %1 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
          iterator_types = ["parallel", "reduction"]}
          ins(%b0 : tensor<?x?xf32>) outs(%b1 : tensor<?xf32>) {
        ^bb1(%bb0 : f32, %bb1 : f32) :
          %2 = arith.addf %bb0, %bb1 : f32
          linalg.yield %2 : f32
      } -> tensor<?xf32>
      iree_linalg_ext.yield %1 : tensor<?xf32>
  } -> tensor<?xf32>
  %3 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
      ins(%0 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
    ^bb0(%b0 : f32, %b1 : f32):
      %4 = arith.mulf %b0, %b0 : f32
      linalg.yield %4 :f32
  } -> tensor<?xf32>
  util.return %3 : tensor<?xf32>
}
