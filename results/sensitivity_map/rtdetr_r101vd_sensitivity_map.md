# Activation divergence: rtdetr / r101vd

4 images; ONNX Runtime `CUDAExecutionProvider`. Relative L2 is measured against the FP32 tensor of the same name.

Higher divergence identifies tensors to inspect; it does not establish that retaining the named node in FP32 improves mAP. Validate any proposed partition with COCO evaluation and a TensorRT build on the target board.

## FP16

Final raw output relative L2: 0.079221

| Node | Op | Tensor | Relative L2 | Mean abs. error | Max abs. error |
| --- | --- | --- | ---: | ---: | ---: |
| `node_softmax_9` | Softmax | `softmax_9` | 0.966095 | 0.00207611 | 0.984776 |
| `node_softmax_1` | Softmax | `softmax_1` | 0.811268 | 0.00241931 | 0.999725 |
| `node_softmax_5` | Softmax | `softmax_5` | 0.778705 | 0.00229314 | 0.999023 |
| `node_softmax_3` | Softmax | `softmax_3` | 0.755651 | 0.00227948 | 0.993588 |
| `node_softmax_7` | Softmax | `softmax_7` | 0.7485 | 0.00231733 | 0.992374 |
| `node_GridSample_4600` | GridSample | `grid_sampler_15` | 0.696383 | 0.0765865 | 1.7127 |
| `node_softmax_11` | Softmax | `softmax_11` | 0.690406 | 0.00171722 | 0.889956 |
| `node_GridSample_4601` | GridSample | `grid_sampler_16` | 0.66284 | 0.0857664 | 2.43372 |
| `node_GridSample_4594` | GridSample | `grid_sampler_9` | 0.651222 | 0.0997046 | 2.82937 |
| `node_GridSample_4597` | GridSample | `grid_sampler_12` | 0.643361 | 0.0942437 | 2.8178 |
| `node_GridSample_4602` | GridSample | `grid_sampler_17` | 0.6274 | 0.0609244 | 1.44344 |
| `node_GridSample_4591` | GridSample | `grid_sampler_6` | 0.624245 | 0.0940651 | 2.65402 |
| `node_GridSample_4588` | GridSample | `grid_sampler_3` | 0.60763 | 0.0952357 | 2.42395 |
| `node_GridSample_4585` | GridSample | `grid_sampler` | 0.602305 | 0.109266 | 2.49012 |
| `node_layer_norm_11` | LayerNormalization | `layer_norm_11` | 0.594565 | 0.212245 | 4.5335 |
| `node_layer_norm_7` | LayerNormalization | `layer_norm_7` | 0.586314 | 0.267406 | 5.36637 |
| `node_layer_norm_8` | LayerNormalization | `layer_norm_8` | 0.580153 | 0.227621 | 7.95952 |
| `node_layer_norm_5` | LayerNormalization | `layer_norm_5` | 0.577086 | 0.194718 | 6.72907 |
| `node_GridSample_4586` | GridSample | `grid_sampler_1` | 0.576127 | 0.107064 | 2.52894 |
| `node_GridSample_4598` | GridSample | `grid_sampler_13` | 0.563266 | 0.0893221 | 2.33478 |
| `node_GridSample_4587` | GridSample | `grid_sampler_2` | 0.560528 | 0.100063 | 3.5782 |
| `node_GridSample_4592` | GridSample | `grid_sampler_7` | 0.539054 | 0.0912344 | 1.98689 |
| `node_layer_norm_4` | LayerNormalization | `layer_norm_4` | 0.536713 | 0.258693 | 6.08847 |
| `node_GridSample_4590` | GridSample | `grid_sampler_5` | 0.526472 | 0.0856621 | 3.14528 |
| `node_GridSample_4595` | GridSample | `grid_sampler_10` | 0.524865 | 0.095103 | 2.28789 |
| `node_layer_norm_10` | LayerNormalization | `layer_norm_10` | 0.520243 | 0.257193 | 6.31263 |
| `node_GridSample_4589` | GridSample | `grid_sampler_4` | 0.516478 | 0.0904703 | 2.06489 |
| `node_layer_norm_14` | LayerNormalization | `layer_norm_14` | 0.510736 | 0.16356 | 4.17319 |
| `node_GridSample_4593` | GridSample | `grid_sampler_8` | 0.505073 | 0.0823349 | 3.88775 |
| `node_layer_norm_13` | LayerNormalization | `layer_norm_13` | 0.502194 | 0.2551 | 11.3259 |
| `node_layer_norm_17` | LayerNormalization | `layer_norm_17` | 0.492646 | 0.141452 | 4.33671 |
| `node_softmax_12` | Softmax | `softmax_12` | 0.479508 | 0.0240251 | 0.93338 |
| `node_GridSample_4599` | GridSample | `grid_sampler_14` | 0.478492 | 0.0629932 | 1.8896 |
| `node_softmax_10` | Softmax | `softmax_10` | 0.461394 | 0.0221208 | 0.825932 |
| `node_layer_norm_18` | LayerNormalization | `layer_norm_18` | 0.457873 | 0.224363 | 8.29013 |
| `node_layer_norm_19` | LayerNormalization | `layer_norm_19` | 0.449627 | 0.222961 | 7.90951 |
| `node_softmax_8` | Softmax | `softmax_8` | 0.448629 | 0.0204954 | 0.985808 |
| `node_layer_norm_16` | LayerNormalization | `layer_norm_16` | 0.446906 | 0.231704 | 5.97779 |
| `node_layer_norm_9` | LayerNormalization | `layer_norm_9` | 0.445426 | 0.140114 | 3.44523 |
| `node_softmax_6` | Softmax | `softmax_6` | 0.432509 | 0.0191142 | 0.863662 |
| `node_layer_norm_20` | LayerNormalization | `layer_norm_20` | 0.427691 | 0.11656 | 4.91278 |
| `node_softmax_4` | Softmax | `softmax_4` | 0.414274 | 0.0182251 | 0.895799 |
| `node_layer_norm_12` | LayerNormalization | `layer_norm_12` | 0.411766 | 0.132206 | 5.319 |
| `node_GridSample_4596` | GridSample | `grid_sampler_11` | 0.409344 | 0.0676228 | 1.80232 |
| `node_layer_norm_6` | LayerNormalization | `layer_norm_6` | 0.40152 | 0.121218 | 3.09301 |
| `node_layer_norm_15` | LayerNormalization | `layer_norm_15` | 0.382124 | 0.140004 | 4.73888 |
| `node_softmax_2` | Softmax | `softmax_2` | 0.358636 | 0.0154097 | 0.645124 |
| `node_layer_norm_3` | LayerNormalization | `layer_norm_3` | 0.223412 | 0.0519953 | 1.48279 |
| `node_softmax` | Softmax | `softmax` | 0.0115108 | 1.24793e-05 | 0.0240929 |
| `node_layer_norm_1` | LayerNormalization | `layer_norm_1` | 0.0066232 | 0.00243883 | 0.622372 |
| `node_layer_norm` | LayerNormalization | `layer_norm` | 0.00632952 | 0.00236722 | 0.371208 |
| `node_layer_norm_2` | LayerNormalization | `layer_norm_2` | 0.00522856 | 0.00131224 | 0.333078 |

## INT8

Final raw output relative L2: 0.120263

| Node | Op | Tensor | Relative L2 | Mean abs. error | Max abs. error |
| --- | --- | --- | ---: | ---: | ---: |
| `node_softmax_9` | Softmax | `softmax_9` | 1.19134 | 0.00393869 | 0.992453 |
| `node_GridSample_4600` | GridSample | `grid_sampler_15` | 1.08818 | 0.182864 | 2.1623 |
| `node_softmax_5` | Softmax | `softmax_5` | 1.085 | 0.0045549 | 0.999143 |
| `node_softmax_1` | Softmax | `softmax_1` | 1.0681 | 0.00465868 | 0.999594 |
| `node_GridSample_4601` | GridSample | `grid_sampler_16` | 1.0619 | 0.21662 | 2.45847 |
| `node_GridSample_4602` | GridSample | `grid_sampler_17` | 1.0468 | 0.163219 | 1.70849 |
| `node_softmax_3` | Softmax | `softmax_3` | 1.04082 | 0.00451598 | 0.991147 |
| `node_softmax_7` | Softmax | `softmax_7` | 1.03377 | 0.00446578 | 0.993827 |
| `node_GridSample_4594` | GridSample | `grid_sampler_9` | 1.0267 | 0.24766 | 2.75512 |
| `node_GridSample_4597` | GridSample | `grid_sampler_12` | 1.00836 | 0.228763 | 3.34287 |
| `node_GridSample_4591` | GridSample | `grid_sampler_6` | 0.994483 | 0.237535 | 2.90342 |
| `node_GridSample_4588` | GridSample | `grid_sampler_3` | 0.981615 | 0.24745 | 2.60814 |
| `node_layer_norm_7` | LayerNormalization | `layer_norm_7` | 0.96857 | 0.719172 | 5.37927 |
| `node_softmax_11` | Softmax | `softmax_11` | 0.968154 | 0.00344234 | 0.954586 |
| `node_layer_norm_11` | LayerNormalization | `layer_norm_11` | 0.957545 | 0.551071 | 4.43908 |
| `node_GridSample_4585` | GridSample | `grid_sampler` | 0.956834 | 0.280925 | 2.57769 |
| `node_layer_norm_5` | LayerNormalization | `layer_norm_5` | 0.952858 | 0.525061 | 7.24005 |
| `node_layer_norm_8` | LayerNormalization | `layer_norm_8` | 0.945981 | 0.604317 | 7.82322 |
| `node_GridSample_4586` | GridSample | `grid_sampler_1` | 0.942636 | 0.28847 | 2.84398 |
| `node_GridSample_4598` | GridSample | `grid_sampler_13` | 0.92441 | 0.235511 | 2.37925 |
| `node_GridSample_4587` | GridSample | `grid_sampler_2` | 0.922548 | 0.270144 | 4.0414 |
| `node_GridSample_4592` | GridSample | `grid_sampler_7` | 0.889663 | 0.245235 | 2.49761 |
| `node_layer_norm_4` | LayerNormalization | `layer_norm_4` | 0.875207 | 0.688842 | 6.87548 |
| `node_GridSample_4590` | GridSample | `grid_sampler_5` | 0.865554 | 0.231256 | 3.59214 |
| `node_GridSample_4595` | GridSample | `grid_sampler_10` | 0.861159 | 0.25197 | 2.4672 |
| `node_GridSample_4589` | GridSample | `grid_sampler_4` | 0.854645 | 0.244571 | 2.28479 |
| `node_layer_norm_10` | LayerNormalization | `layer_norm_10` | 0.850583 | 0.678115 | 7.09509 |
| `node_GridSample_4593` | GridSample | `grid_sampler_8` | 0.824215 | 0.221206 | 4.136 |
| `node_layer_norm_13` | LayerNormalization | `layer_norm_13` | 0.812846 | 0.659035 | 10.4821 |
| `node_layer_norm_14` | LayerNormalization | `layer_norm_14` | 0.807159 | 0.410365 | 5.96429 |
| `node_GridSample_4599` | GridSample | `grid_sampler_14` | 0.803024 | 0.171861 | 2.08789 |
| `node_softmax_12` | Softmax | `softmax_12` | 0.779835 | 0.0651357 | 0.91228 |
| `node_softmax_10` | Softmax | `softmax_10` | 0.776034 | 0.0607856 | 0.800304 |
| `node_layer_norm_17` | LayerNormalization | `layer_norm_17` | 0.7675 | 0.346011 | 7.44643 |
| `node_layer_norm_9` | LayerNormalization | `layer_norm_9` | 0.75304 | 0.383951 | 3.51493 |
| `node_softmax_8` | Softmax | `softmax_8` | 0.737851 | 0.055242 | 0.990285 |
| `node_layer_norm_18` | LayerNormalization | `layer_norm_18` | 0.734036 | 0.571989 | 12.635 |
| `node_layer_norm_16` | LayerNormalization | `layer_norm_16` | 0.730645 | 0.591082 | 9.51884 |
| `node_softmax_4` | Softmax | `softmax_4` | 0.720185 | 0.052108 | 0.897385 |
| `node_layer_norm_19` | LayerNormalization | `layer_norm_19` | 0.710966 | 0.558109 | 13.2941 |
| `node_softmax_6` | Softmax | `softmax_6` | 0.702816 | 0.0515764 | 0.966272 |
| `node_layer_norm_6` | LayerNormalization | `layer_norm_6` | 0.698024 | 0.343984 | 3.17077 |
| `node_GridSample_4596` | GridSample | `grid_sampler_11` | 0.694757 | 0.186056 | 2.15961 |
| `node_layer_norm_12` | LayerNormalization | `layer_norm_12` | 0.68444 | 0.354329 | 4.95934 |
| `node_layer_norm_20` | LayerNormalization | `layer_norm_20` | 0.651485 | 0.281645 | 5.88746 |
| `node_layer_norm_15` | LayerNormalization | `layer_norm_15` | 0.647148 | 0.37931 | 5.9393 |
| `node_softmax_2` | Softmax | `softmax_2` | 0.63943 | 0.0448566 | 0.602852 |
| `node_layer_norm_3` | LayerNormalization | `layer_norm_3` | 0.397611 | 0.155283 | 1.70257 |
| `node_layer_norm_2` | LayerNormalization | `layer_norm_2` | 0.285213 | 0.142027 | 2.13037 |
| `node_softmax` | Softmax | `softmax` | 0.185742 | 0.000411355 | 0.116992 |
| `node_layer_norm_1` | LayerNormalization | `layer_norm_1` | 0.161705 | 0.126954 | 2.34729 |
| `node_layer_norm` | LayerNormalization | `layer_norm` | 0.151174 | 0.114374 | 2.10017 |

