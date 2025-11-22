module {
  util.func public @captureDims(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
    %c1 = arith.constant 1 : index
    %0 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1, %arg3, %arg2, %arg4) : (tensor<?x?xf32>{%arg1, %arg2}, index, index, index, index) -> tensor<?x?xf32>{%arg3, %arg4} =
        (%arg5: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
      %1 = flow.dispatch.tie_shape %arg5 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg6, %arg8}
      %2 = flow.dispatch.tie_shape %arg10 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg7, %arg9}
      flow.return
    }
    util.return
  }
  util.func public @capture2DimsForOneTensor(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index, %arg3: index, %arg4: index) {
    %c1 = arith.constant 1 : index
    %0 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<?x?xf32>{%arg1, %arg2}, index, index, index, index) -> tensor<?x?xf32>{%arg3, %arg4} =
        (%arg5: !flow.dispatch.tensor<readonly:tensor<?x?xf32>>, %arg6: index, %arg7: index, %arg8: index, %arg9: index, %arg10: !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>) {
      %1 = flow.dispatch.tie_shape %arg5 : !flow.dispatch.tensor<readonly:tensor<?x?xf32>>{%arg6, %arg7}
      %2 = flow.dispatch.tie_shape %arg10 : !flow.dispatch.tensor<writeonly:tensor<?x?xf32>>{%arg8, %arg9}
      flow.return
    }
    util.return
  }
  util.func public @capturedTiedDims(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) {
    %c1 = arith.constant 1 : index
    %0 = flow.dispatch.workgroups[%c1, %c1, %c1](%arg0, %arg1, %arg2) : (tensor<?x?xf32>{%arg1, %arg2}, index, index) -> %arg0{%arg1, %arg2} =
        (%arg3: !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>, %arg4: index, %arg5: index) {
      %1 = flow.dispatch.tie_shape %arg3 : !flow.dispatch.tensor<readwrite:tensor<?x?xf32>>{%arg4, %arg5}
      flow.return
    }
    util.return
  }
}

