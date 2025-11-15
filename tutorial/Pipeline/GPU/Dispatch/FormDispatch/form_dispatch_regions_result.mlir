#map = affine_map<()[s0] -> (s0 ceildiv 8)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @pack_elementwise_fusion(%arg0: tensor<?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = affine.apply #map()[%dim]
    %2 = affine.apply #map1()[%dim_0]
    %3 = tensor.empty(%1, %2) : tensor<?x?x8x32xf32>
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %4 = flow.dispatch.region -> (tensor<?x?x8x32xf32>{%1, %2}) {
      %5 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1, %arg0 : tensor<?x?xf32>, tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %6 = arith.addf %in, %in_3 : f32
        linalg.yield %6 : f32
      } -> tensor<?x?xf32>
      %pack = tensor.pack %5 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %3 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
      flow.return %pack : tensor<?x?x8x32xf32>
    }
    util.return %4 : tensor<?x?x8x32xf32>
  }
}

// -----
#map = affine_map<()[s0] -> (s0 ceildiv 8)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 32)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @pack_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %3 = affine.apply #map()[%dim]
    %4 = affine.apply #map1()[%dim_0]
    %5 = tensor.empty(%3, %4) : tensor<?x?x8x32xf32>
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %6 = flow.dispatch.region -> (tensor<?x?x8x32xf32>{%3, %4}) {
      %7 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %9 = arith.addf %in, %out : f32
        linalg.yield %9 : f32
      } -> tensor<?xf32>
      %8 = linalg.generic {indexing_maps = [#map2, #map3, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg1, %7 : tensor<?x?xf32>, tensor<?xf32>) outs(%2 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %9 = arith.addf %in, %in_3 : f32
        linalg.yield %9 : f32
      } -> tensor<?x?xf32>
      %pack = tensor.pack %8 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %5 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
      flow.return %pack : tensor<?x?x8x32xf32>
    }
    util.return %6 : tensor<?x?x8x32xf32>
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<()[s0] -> (s0 ceildiv 8)>
#map3 = affine_map<()[s0] -> (s0 ceildiv 32)>
module {
  util.func public @tranpose_pack_fusion(%arg0: tensor<?x?xf32>) -> tensor<?x?x8x32xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %1 = flow.dispatch.region -> (tensor<?x?xf32>{%dim, %dim_0}) {
      %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?xf32>
      flow.return %6 : tensor<?x?xf32>
    }
    %2 = affine.apply #map2()[%dim]
    %3 = affine.apply #map3()[%dim_0]
    %4 = tensor.empty(%2, %3) : tensor<?x?x8x32xf32>
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %5 = flow.dispatch.region -> (tensor<?x?x8x32xf32>{%2, %3}) {
      %pack = tensor.pack %1 padding_value(%cst : f32) inner_dims_pos = [0, 1] inner_tiles = [8, 32] into %4 : tensor<?x?xf32> -> tensor<?x?x8x32xf32>
      flow.return %pack : tensor<?x?x8x32xf32>
    }
    util.return %5 : tensor<?x?x8x32xf32>
  }
}

// -----
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @set_encoding_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: index, %arg3: index) -> tensor<?x?xf32, #encoding> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %3 = flow.dispatch.region -> (tensor<?x?xf32, #encoding>{%dim, %dim_0}) {
      %4 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %7 = arith.addf %in, %out : f32
        linalg.yield %7 : f32
      } -> tensor<?xf32>
      %5 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%arg1, %4 : tensor<?x?xf32>, tensor<?xf32>) outs(%2 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %7 = arith.addf %in, %in_3 : f32
        linalg.yield %7 : f32
      } -> tensor<?x?xf32>
      %6 = iree_encoding.set_encoding %5 : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
      flow.return %6 : tensor<?x?xf32, #encoding>
    }
    util.return %3 : tensor<?x?xf32, #encoding>
  }
}

// -----
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#map = affine_map<()[s0, s1] -> (s0 + s1)>
module {
  util.func public @set_encoding_pad_fusion(%arg0: tensor<?x?xf32>, %arg1: index, %arg2: index) -> tensor<?x?xf32, #encoding> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %0 = affine.apply #map()[%arg1, %dim]
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %1 = affine.apply #map()[%arg2, %dim_0]
    %c0_1 = arith.constant 0 : index
    %dim_2 = tensor.dim %arg0, %c0_1 : tensor<?x?xf32>
    %2 = affine.apply #map()[%arg1, %dim_2]
    %c1_3 = arith.constant 1 : index
    %dim_4 = tensor.dim %arg0, %c1_3 : tensor<?x?xf32>
    %3 = affine.apply #map()[%arg2, %dim_4]
    %c0_5 = arith.constant 0 : index
    %c1_6 = arith.constant 1 : index
    %4 = flow.dispatch.region -> (tensor<?x?xf32, #encoding>{%0, %3}) {
      %padded = tensor.pad %arg0 low[0, 0] high[%arg1, %arg2] {
      ^bb0(%arg3: index, %arg4: index):
        tensor.yield %cst : f32
      } : tensor<?x?xf32> to tensor<?x?xf32>
      %5 = iree_encoding.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
      flow.return %5 : tensor<?x?xf32, #encoding>
    }
    util.return %4 : tensor<?x?xf32, #encoding>
  }
}

// -----
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#map = affine_map<()[s0, s1] -> (s0 + s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @set_encoding_pad_elementwise_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: index, %arg3: index) -> tensor<?x?xf32, #encoding> {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = tensor.empty(%dim) : tensor<?xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<?xf32>) -> tensor<?xf32>
    %2 = tensor.empty(%dim, %dim_0) : tensor<?x?xf32>
    %c0_1 = arith.constant 0 : index
    %3 = affine.apply #map()[%arg2, %dim]
    %c1_2 = arith.constant 1 : index
    %4 = affine.apply #map()[%arg3, %dim_0]
    %c0_3 = arith.constant 0 : index
    %5 = affine.apply #map()[%arg2, %dim]
    %c1_4 = arith.constant 1 : index
    %6 = affine.apply #map()[%arg3, %dim_0]
    %c0_5 = arith.constant 0 : index
    %c1_6 = arith.constant 1 : index
    %7 = flow.dispatch.region -> (tensor<?x?xf32, #encoding>{%3, %6}) {
      %8 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%1 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %11 = arith.addf %in, %out : f32
        linalg.yield %11 : f32
      } -> tensor<?xf32>
      %9 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %8 : tensor<?x?xf32>, tensor<?xf32>) outs(%2 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_7: f32, %out: f32):
        %11 = arith.addf %in, %in_7 : f32
        linalg.yield %11 : f32
      } -> tensor<?x?xf32>
      %padded = tensor.pad %9 low[0, 0] high[%arg2, %arg3] {
      ^bb0(%arg4: index, %arg5: index):
        tensor.yield %cst : f32
      } : tensor<?x?xf32> to tensor<?x?xf32>
      %10 = iree_encoding.set_encoding %padded : tensor<?x?xf32> -> tensor<?x?xf32, #encoding>
      flow.return %10 : tensor<?x?xf32, #encoding>
    }
    util.return %7 : tensor<?x?xf32, #encoding>
  }
}

// -----
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @unset_encoding_elementwise_fusion(%arg0: tensor<?x?xf32, #encoding>, %arg1: tensor<?xf32>, %arg2: index, %arg3: index) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
    %1 = flow.dispatch.region -> (tensor<?x?xf32>{%arg2, %arg3}) {
      %2 = iree_encoding.unset_encoding %arg0 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%arg2, %arg3}
      %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %4 = arith.addf %in, %in_0 : f32
        linalg.yield %4 : f32
      } -> tensor<?x?xf32>
      flow.return %3 : tensor<?x?xf32>
    }
    util.return %1 : tensor<?x?xf32>
  }
}

// -----
#encoding = #iree_encoding.encoding<operand_index = 0 : i64, op_type =  matmul, element_types = [f32, f32, f32]>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @unset_encoding_slice_elementwise_fusion(%arg0: tensor<?x?xf32, #encoding>, %arg1: tensor<?xf32>, %arg2: index, %arg3: index) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = tensor.empty(%arg2, %arg3) : tensor<?x?xf32>
    %1 = flow.dispatch.region -> (tensor<?x?xf32>{%arg2, %arg3}) {
      %2 = iree_encoding.unset_encoding %arg0 : tensor<?x?xf32, #encoding> -> tensor<?x?xf32>{%arg2, %arg3}
      %extracted_slice = tensor.extract_slice %2[0, 0] [%arg2, %arg3] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
      %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%extracted_slice, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %4 = arith.addf %in, %in_0 : f32
        linalg.yield %4 : f32
      } -> tensor<?x?xf32>
      flow.return %3 : tensor<?x?xf32>
    }
    util.return %1 : tensor<?x?xf32>
  }
}

// -----
#map = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @unpack_elementwise_fusion(%arg0: tensor<?x?x?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?x?xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?x?xf32>
    %dim_2 = tensor.dim %arg0, %c3 : tensor<?x?x?x?xf32>
    %0 = affine.apply #map()[%dim, %dim_1]
    %1 = affine.apply #map()[%dim_0, %dim_2]
    %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
    %3 = flow.dispatch.region -> (tensor<?x?xf32>{%0, %1}) {
      %unpack = tensor.unpack %arg0 inner_dims_pos = [0, 1] inner_tiles = [%dim_1, %dim_2] into %2 : tensor<?x?x?x?xf32> -> tensor<?x?xf32>
      %4 = linalg.generic {indexing_maps = [#map1, #map2, #map1], iterator_types = ["parallel", "parallel"]} ins(%unpack, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%2 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %in_3: f32, %out: f32):
        %5 = arith.addf %in, %in_3 : f32
        linalg.yield %5 : f32
      } -> tensor<?x?xf32>
      flow.return %4 : tensor<?x?xf32>
    }
    util.return %3 : tensor<?x?xf32>
  }
}

// -----
#map = affine_map<()[s0, s1] -> (s0 * s1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @unpack_non_intersecting_reduction(%arg0: tensor<?x?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?x?xf32>
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?x?xf32>
    %dim_1 = tensor.dim %arg0, %c2 : tensor<?x?x?xf32>
    %0 = affine.apply #map()[%dim_0, %dim_1]
    %1 = tensor.empty(%dim, %0) : tensor<?x?xf32>
    %2 = tensor.empty(%0) : tensor<?xf32>
    %3 = flow.dispatch.region -> (tensor<?xf32>{%0}) {
      %unpack = tensor.unpack %arg0 inner_dims_pos = [1] inner_tiles = [%dim_1] into %1 : tensor<?x?x?xf32> -> tensor<?x?xf32>
      %4 = linalg.generic {indexing_maps = [#map1, #map2, #map2], iterator_types = ["parallel", "reduction"]} ins(%unpack, %arg1 : tensor<?x?xf32>, tensor<?xf32>) outs(%2 : tensor<?xf32>) {
      ^bb0(%in: f32, %in_2: f32, %out: f32):
        %5 = arith.addf %in, %in_2 : f32
        %6 = arith.addf %5, %out : f32
        linalg.yield %6 : f32
      } -> tensor<?xf32>
      flow.return %4 : tensor<?xf32>
    }
    util.return %3 : tensor<?xf32>
  }
}

// -----
#map = affine_map<()[s0, s1, s2] -> ((s1 - s0) ceildiv s2)>
#map1 = affine_map<(d0, d1) -> ()>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  util.func public @data_dependent_shape(%arg0: tensor<f32>, %arg1: tensor<2xi32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %extracted = tensor.extract %arg1[%c0] : tensor<2xi32>
    %0 = arith.index_cast %extracted : i32 to index
    %extracted_0 = tensor.extract %arg1[%c1] : tensor<2xi32>
    %1 = arith.index_cast %extracted_0 : i32 to index
    %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
    %c0_1 = arith.constant 0 : index
    %c1_2 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c0_4 = arith.constant 0 : index
    %c1_5 = arith.constant 1 : index
    %3 = affine.apply #map()[%c0_4, %0, %c1_5]
    %c0_6 = arith.constant 0 : index
    %c1_7 = arith.constant 1 : index
    %4 = affine.apply #map()[%c0_6, %1, %c1_7]
    %5 = flow.dispatch.region[%3, %4] -> (tensor<?x?xf32>{%0, %1}) {
      %6 = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<f32>) outs(%2 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        linalg.yield %in : f32
      } -> tensor<?x?xf32>
      flow.return %6 : tensor<?x?xf32>
    } count(%arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg2, %arg3
      flow.return %x, %y, %z : index, index, index
    }
    util.return %5 : tensor<?x?xf32>
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @no_yield_dead_results(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg2, %c0 : tensor<?xf32>
    %0 = flow.dispatch.region -> (tensor<?xf32>{%dim}) {
      %1:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg1, %arg2 : tensor<?xf32>, tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32, %out_0: f32):
        %2 = arith.addf %in, %out : f32
        %3 = arith.addf %in, %out_0 : f32
        linalg.yield %2, %3 : f32, f32
      } -> (tensor<?xf32>, tensor<?xf32>)
      flow.return %1#1 : tensor<?xf32>
    }
    util.return %0 : tensor<?xf32>
  }
}

// -----
#map = affine_map<(d0) -> (d0)>
module {
  util.func public @scf_nested_dispatch(%arg0: tensor<?xi32>) -> tensor<?xi32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xi32>
    %0 = tensor.empty(%dim) : tensor<?xi32>
    %1 = arith.cmpi eq, %dim, %c1 : index
    %2 = scf.if %1 -> (tensor<?xi32>) {
      %3 = flow.dispatch.region -> (tensor<?xi32>{%dim}) {
        %4 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : tensor<?xi32>) outs(%0 : tensor<?xi32>) {
        ^bb0(%in: i32, %out: i32):
          %5 = arith.addi %in, %in : i32
          linalg.yield %5 : i32
        } -> tensor<?xi32>
        flow.return %4 : tensor<?xi32>
      }
      scf.yield %3 : tensor<?xi32>
    } else {
      scf.yield %arg0 : tensor<?xi32>
    }
    util.return %2 : tensor<?xi32>
  }
}

// -----
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  util.func public @no_dequantization_fusion(%arg0: tensor<4096x32x128xi8>, %arg1: tensor<1x1x32x128xf32>, %arg2: tensor<4096x32xf32>, %arg3: tensor<4096x32xf32>) -> tensor<1x1x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x1x4096xf32>
    %1 = tensor.empty() : tensor<4096x32x128xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x1x4096xf32>) -> tensor<1x1x4096xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map1, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg2, %arg3 : tensor<4096x32x128xi8>, tensor<4096x32xf32>, tensor<4096x32xf32>) outs(%1 : tensor<4096x32x128xf32>) {
    ^bb0(%in: i8, %in_0: f32, %in_1: f32, %out: f32):
      %5 = arith.extui %in : i8 to i32
      %6 = arith.uitofp %5 : i32 to f32
      %7 = arith.subf %6, %in_1 : f32
      %8 = arith.mulf %7, %in_0 : f32
      linalg.yield %8 : f32
    } -> tensor<4096x32x128xf32>
    %4 = flow.dispatch.region -> (tensor<1x1x4096xf32>) {
      %5 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%arg1, %3 : tensor<1x1x32x128xf32>, tensor<4096x32x128xf32>) outs(%2 : tensor<1x1x4096xf32>) {
      ^bb0(%in: f32, %in_0: f32, %out: f32):
        %6 = arith.mulf %in, %in_0 : f32
        %7 = arith.addf %6, %out : f32
        linalg.yield %7 : f32
      } -> tensor<1x1x4096xf32>
      flow.return %5 : tensor<1x1x4096xf32>
    }
    util.return %4 : tensor<1x1x4096xf32>
  }
}

// -----
#map = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
module {
  util.func public @no_dequantization_like_fusion(%arg0: tensor<32x1x16x1x8xi16>, %arg1: tensor<32x344x16x32x8xi4>) -> tensor<32x1x344x1x32xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = tensor.empty() : tensor<32x1x16x1x8xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<32x1x16x1x8xi16>) outs(%0 : tensor<32x1x16x1x8xi32>) {
    ^bb0(%in: i16, %out: i32):
      %7 = arith.extsi %in : i16 to i32
      linalg.yield %7 : i32
    } -> tensor<32x1x16x1x8xi32>
    %2 = tensor.empty() : tensor<32x344x16x32x8xi32>
    %3 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%arg1 : tensor<32x344x16x32x8xi4>) outs(%2 : tensor<32x344x16x32x8xi32>) {
    ^bb0(%in: i4, %out: i32):
      %7 = arith.extui %in : i4 to i32
      linalg.yield %7 : i32
    } -> tensor<32x344x16x32x8xi32>
    %4 = tensor.empty() : tensor<32x1x344x1x32xi32>
    %5 = linalg.fill ins(%c0_i32 : i32) outs(%4 : tensor<32x1x344x1x32xi32>) -> tensor<32x1x344x1x32xi32>
    %6 = flow.dispatch.region -> (tensor<32x1x344x1x32xi32>) {
      %7 = linalg.batch_mmt4d ins(%1, %3 : tensor<32x1x16x1x8xi32>, tensor<32x344x16x32x8xi32>) outs(%5 : tensor<32x1x344x1x32xi32>) -> tensor<32x1x344x1x32xi32>
      flow.return %7 : tensor<32x1x344x1x32xi32>
    }
    util.return %6 : tensor<32x1x344x1x32xi32>
  }
}

// -----
#map = affine_map<(d0, d1, d2) -> (d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  util.func public @broadcasting_dequant_op(%arg0: tensor<?x?xi8>, %arg1: tensor<?x?x?xi32>, %arg2: tensor<?x?x?xi32>) -> tensor<?x?x?xi32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg2, %c0 : tensor<?x?x?xi32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg2, %c1 : tensor<?x?x?xi32>
    %c2 = arith.constant 2 : index
    %dim_1 = tensor.dim %arg2, %c2 : tensor<?x?x?xi32>
    %c0_2 = arith.constant 0 : index
    %c1_3 = arith.constant 1 : index
    %dim_4 = tensor.dim %arg0, %c0_2 : tensor<?x?xi8>
    %dim_5 = tensor.dim %arg0, %c1_3 : tensor<?x?xi8>
    %dim_6 = tensor.dim %arg1, %c0_2 : tensor<?x?x?xi32>
    %0 = tensor.empty(%dim_6, %dim_4, %dim_5) : tensor<?x?x?xi32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<?x?xi8>) outs(%0 : tensor<?x?x?xi32>) {
    ^bb0(%in: i8, %out: i32):
      %3 = arith.extui %in : i8 to i32
      linalg.yield %3 : i32
    } -> tensor<?x?x?xi32>
    %2 = flow.dispatch.region -> (tensor<?x?x?xi32>{%dim, %dim_0, %dim_1}) {
      %3 = linalg.batch_matmul_transpose_b ins(%1, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xi32>) outs(%arg2 : tensor<?x?x?xi32>) -> tensor<?x?x?xi32>
      flow.return %3 : tensor<?x?x?xi32>
    }
    util.return %2 : tensor<?x?x?xi32>
  }
}

// -----
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
module {
  util.func public @softmax_like_fusion(%arg0: tensor<2x4096x640xf16>, %arg1: tensor<640xf16>, %arg2: tensor<640xf16>) -> tensor<2x4096x640x1xf16> {
    %expanded = tensor.expand_shape %arg0 [[0], [1], [2, 3]] output_shape [2, 4096, 640, 1] : tensor<2x4096x640xf16> into tensor<2x4096x640x1xf16>
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.100000e+01 : f32
    %cst_1 = arith.constant 4.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x4096x640xf32>
    %1 = tensor.empty() : tensor<2x4096x640x1xf16>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x4096x640xf16>) outs(%0 : tensor<2x4096x640xf32>) {
    ^bb0(%in: f16, %out: f32):
      %6 = arith.extf %in : f16 to f32
      linalg.yield %6 : f32
    } -> tensor<2x4096x640xf32>
    %3 = tensor.empty() : tensor<2x4096xf32>
    %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<2x4096xf32>) -> tensor<2x4096xf32>
    %expanded_2 = tensor.expand_shape %arg1 [[0, 1]] output_shape [640, 1] : tensor<640xf16> into tensor<640x1xf16>
    %expanded_3 = tensor.expand_shape %arg2 [[0, 1]] output_shape [640, 1] : tensor<640xf16> into tensor<640x1xf16>
    %5 = flow.dispatch.region -> (tensor<2x4096x640x1xf16>) {
      %6 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2 : tensor<2x4096x640xf32>) outs(%4 : tensor<2x4096xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.addf %in, %out : f32
        linalg.yield %10 : f32
      } -> tensor<2x4096xf32>
      %7 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<2x4096xf32>) outs(%3 : tensor<2x4096xf32>) {
      ^bb0(%in: f32, %out: f32):
        %10 = arith.divf %in, %cst_0 : f32
        linalg.yield %10 : f32
      } -> tensor<2x4096xf32>
      %8 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "reduction"]} ins(%2, %7 : tensor<2x4096x640xf32>, tensor<2x4096xf32>) outs(%4 : tensor<2x4096xf32>) {
      ^bb0(%in: f32, %in_4: f32, %out: f32):
        %10 = arith.subf %in, %in_4 : f32
        %11 = arith.mulf %10, %10 : f32
        %12 = arith.addf %11, %out : f32
        linalg.yield %12 : f32
      } -> tensor<2x4096xf32>
      %9 = linalg.generic {indexing_maps = [#map3, #map4, #map4, #map5, #map5, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded, %7, %8, %expanded_2, %expanded_3 : tensor<2x4096x640x1xf16>, tensor<2x4096xf32>, tensor<2x4096xf32>, tensor<640x1xf16>, tensor<640x1xf16>) outs(%1 : tensor<2x4096x640x1xf16>) {
      ^bb0(%in: f16, %in_4: f32, %in_5: f32, %in_6: f16, %in_7: f16, %out: f16):
        %10 = arith.divf %in_5, %cst_0 : f32
        %11 = arith.addf %10, %cst_1 : f32
        %12 = math.rsqrt %11 : f32
        %13 = arith.extf %in : f16 to f32
        %14 = arith.subf %13, %in_4 : f32
        %15 = arith.mulf %14, %12 : f32
        %16 = arith.extf %in_6 : f16 to f32
        %17 = arith.mulf %15, %16 : f32
        %18 = arith.extf %in_7 : f16 to f32
        %19 = arith.addf %17, %18 : f32
        %20 = arith.truncf %19 : f32 to f16
        linalg.yield %20 : f16
      } -> tensor<2x4096x640x1xf16>
      flow.return %9 : tensor<2x4096x640x1xf16>
    }
    util.return %5 : tensor<2x4096x640x1xf16>
  }
}

// -----
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

// -----
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0) -> (d0)>
module {
  util.func public @custom_op_consumer_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
    %0 = flow.dispatch.region -> (tensor<?xf32>{%dim}) {
      %1 = iree_linalg_ext.custom_op{indexing_maps = [#map, #map1], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]} ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%arg2: tensor<?x?xf32>, %arg3: tensor<?xf32>):
        %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x?xf32>) outs(%arg3 : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.addf %in, %out : f32
          linalg.yield %4 : f32
        } -> tensor<?xf32>
        iree_linalg_ext.yield %3 : tensor<?xf32>
      } -> tensor<?xf32>
      %2 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel"]} ins(%1 : tensor<?xf32>) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.mulf %in, %in : f32
        linalg.yield %3 : f32
      } -> tensor<?xf32>
      flow.return %2 : tensor<?xf32>
    }
    util.return %0 : tensor<?xf32>
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
module {
  util.func public @custom_op_producer_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg1, %c0 : tensor<?xf32>
    %0 = flow.dispatch.region -> (tensor<?xf32>{%dim}) {
      %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.mulf %in, %in : f32
        linalg.yield %3 : f32
      } -> tensor<?x?xf32>
      %2 = iree_linalg_ext.custom_op{indexing_maps = [#map, #map1], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<reduction>]} ins(%1 : tensor<?x?xf32>) outs(%arg1 : tensor<?xf32>) {
      ^bb0(%arg2: tensor<?x?xf32>, %arg3: tensor<?xf32>):
        %3 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "reduction"]} ins(%arg2 : tensor<?x?xf32>) outs(%arg3 : tensor<?xf32>) {
        ^bb0(%in: f32, %out: f32):
          %4 = arith.addf %in, %out : f32
          linalg.yield %4 : f32
        } -> tensor<?xf32>
        iree_linalg_ext.yield %3 : tensor<?xf32>
      } -> tensor<?xf32>
      flow.return %2 : tensor<?xf32>
    }
    util.return %0 : tensor<?xf32>
  }
}

// -----
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1)[s0, s1] -> (d0, s0)>
#map2 = affine_map<(d0, d1)[s0, s1] -> (s0, s1)>
#map3 = affine_map<(d0, d1)[s0, s1] -> (d0, s1)>
#map4 = affine_map<(d0, d1)[s0, s1] -> (s1, d1)>
#map5 = affine_map<(d0, d1)[s0, s1] -> (d0, d1)>
module {
  util.func public @custom_op_no_producer_fusion(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32>, %arg3: tensor<?x?xf32>, %arg4: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %0 = flow.dispatch.region -> (tensor<?x?xf32>{%dim, %dim_0}) {
      %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<?x?xf32>) outs(%arg0 : tensor<?x?xf32>) {
      ^bb0(%in: f32, %out: f32):
        %3 = arith.mulf %in, %in : f32
        linalg.yield %3 : f32
      } -> tensor<?x?xf32>
      flow.return %2 : tensor<?x?xf32>
    }
    %c0_1 = arith.constant 0 : index
    %dim_2 = tensor.dim %arg4, %c0_1 : tensor<?x?xf32>
    %c1_3 = arith.constant 1 : index
    %dim_4 = tensor.dim %arg4, %c1_3 : tensor<?x?xf32>
    %1 = flow.dispatch.region -> (tensor<?x?xf32>{%dim_2, %dim_4}) {
      %2 = iree_linalg_ext.custom_op{indexing_maps = [#map1, #map2, #map3, #map4, #map5], iterator_types = [#iree_linalg_ext.iterator_type<parallel>, #iree_linalg_ext.iterator_type<parallel>]} ins(%0, %arg1, %arg2, %arg3 : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg4 : tensor<?x?xf32>) {
      ^bb0(%arg5: tensor<?x?xf32>, %arg6: tensor<?x?xf32>, %arg7: tensor<?x?xf32>, %arg8: tensor<?x?xf32>, %arg9: tensor<?x?xf32>):
        %3 = linalg.matmul ins(%arg5, %arg6 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg7 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %4 = linalg.matmul ins(%3, %arg8 : tensor<?x?xf32>, tensor<?x?xf32>) outs(%arg9 : tensor<?x?xf32>) -> tensor<?x?xf32>
        iree_linalg_ext.yield %4 : tensor<?x?xf32>
      } -> tensor<?x?xf32>
      flow.return %2 : tensor<?x?xf32>
    }
    util.return %1 : tensor<?x?xf32>
  }
}

