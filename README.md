# Content

## dataset.py
Customised CIFAR-10 Dataset for minimal running. Accepts `dual_input` parameter to enable dual image output. Note that images are identically duplicated only.

## models.py
1. VGG16 series
2. VGG11 series
3. ReducedVGG11 series
4. STCNN (Spatial Transformer CNN) series

Each series has `Dual` or not version.  
`VGG` series mainly exists for basic pipeline purpose and experimentation.  
`STCNN` implements from the paper [Spatial Transformer Networks](http://papers.neurips.cc/paper/5854-spatial-transformer-networks.pdf), which adds spatial invariance (any affine transformation such as rotation, scaling, shearing) property to the network. Spatial Transformer blocks are easily pluggable to any neural network.

## main.py
A basic pipeline that is almost self-explanatory. It has "setting-train-testing-save" workflow, but normally only "setting-testing" is used. Settings are minimal, no hyper-parameter optimisation. This serves as template for any expansion to a real experiment.

The arguments are inside `run()` and encouraged to modify variables in the space provided at the beginning of the function, instead of wrapping them as arguments of the function.

# To Improve
- No suitable dataset
- Not trained models
- Not exhaustive review