# ML-topology
Solving Chern numbers using convolutional neural nets (used in the manuscript "RG-Inspired Neural Networks for Computing Topological Invariants," arxiv.org/abs/2202.07669)

The MATLAB code generates lattice samples, while the Python code defines, trains, and evaluates the neural networks. Data used for Fig. 1(b), (d), and (f) are available as MATLAB data files data_1.mat, data_2.mat, and data_3.mat.

To generate samples (4x4, 8x8, or 16x16), run generate_lattice_samples_disorder.m. Set the flags in the SETTINGS section and, if using different lattice parameters, set those in the LATTICE PARAMETERS section. The other MATLAB scripts are referenced and should be placed in the same directory.

The base ResNet is created and trained in Base_ResNet.py. The RG network is created and initialized in RG_ResNet_initialize_to_decimation.py (save it manually after sufficient testing), and then the initialized network is loaded and trained on Chern labels in RG_ResNet_save_intermediate_models.py. Custom losses, functions, and layers are in custom_functions.py.

Evaluation of 8x8 and 16x16 accuracy is done by get_scores.py. Plots are made by explore_RG.py and phase_diagram_4x4.py.
