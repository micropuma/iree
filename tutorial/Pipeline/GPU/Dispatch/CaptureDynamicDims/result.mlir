#map = affine_map<()[s0, s1, s2] -> ((s1 - s0) ceildiv s2)>
#map1 = affine_map<(d0) -> (d0)>
module {
  func.func @test(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = hal.buffer_view.dim<%arg0 : !hal.buffer_view>[0] : index
    %1 = hal.tensor.import %arg0 : !hal.buffer_view -> tensor<?xf32>{%0}
    %2 = hal.buffer_view.dim<%arg1 : !hal.buffer_view>[0] : index
    %3 = hal.tensor.import %arg1 : !hal.buffer_view -> tensor<?xf32>{%2}
    %4 = affine.apply #map()[%c0, %0, %c1]
    %5 = flow.dispatch.workgroups[%4](%0, %1, %3, %0, %2, %0) : (index, tensor<?xf32>{%0}, tensor<?xf32>{%2}, index, index, index) -> tensor<?xf32>{%0} =
        (%arg2: index, %arg3: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg4: !flow.dispatch.tensor<readonly:tensor<?xf32>>, %arg5: index, %arg6: index, %arg7: index, %arg8: !flow.dispatch.tensor<writeonly:tensor<?xf32>>) {
      %7 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg7}
      %8 = flow.dispatch.tie_shape %arg4 : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg6}
      %9 = flow.dispatch.tie_shape %arg8 : !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg7}
      %10 = flow.dispatch.tensor.load %7, offsets = [0], sizes = [%arg7], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg7} -> tensor<?xf32>
      %11 = flow.dispatch.tensor.load %8, offsets = [0], sizes = [%arg6], strides = [1] : !flow.dispatch.tensor<readonly:tensor<?xf32>>{%arg6} -> tensor<?xf32>
      %12 = tensor.empty(%arg7) : tensor<?xf32>
      %13 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel"]} ins(%10, %11 : tensor<?xf32>, tensor<?xf32>) outs(%12 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %14 = arith.addf %in, %in_0 : f32
        linalg.yield %14 : f32
      } -> tensor<?xf32>
      flow.dispatch.tensor.store %13, %9, offsets = [0], sizes = [%arg7], strides = [1] : tensor<?xf32> -> !flow.dispatch.tensor<writeonly:tensor<?xf32>>{%arg7}
      flow.return
    } count(%arg2: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg2
      flow.return %x, %y, %z : index, index, index
    }
    %6 = hal.tensor.export %5 : tensor<?xf32>{%0} -> !hal.buffer_view
    return %6 : !hal.buffer_view
  }
}

