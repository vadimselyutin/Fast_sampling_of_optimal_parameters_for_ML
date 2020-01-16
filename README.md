# Fast_sampling_of_optimal_parameters_for_ML
Please read project_report.pdf for more info.

## Skoltech NLA final project

### Time tests of the algorithm for neural networks with different number of weights
https://colab.research.google.com/drive/1FGEjZ_OG-F9EXEBE9bOk2tPx-a5gj26c

### Model predictions with generated parameters
https://colab.research.google.com/drive/1EVVzbXVBpwpAM-K2v96c-cWQNI7FcTnl

### slq_upd.py
Lanczos method itself is implemented in this module. The key function _lanczos_m_upd computes symmetric m × m tridiagonal matrix T and matrix V with orthogonal rows constituting the basis of the Krylov subspace Km (A, x), where x is an arbitrary starting unit vector.

### lanczos_upd.py
The main purpose of lanczos_upd.py is to transform the input ModelHessianOperator object to the ScipyLinearOperator object and then call the _lanczos_m_upd function.

### hvp_operator_upd.py
This is the key script to be launched for the computations. In this module first we define a linear operator Model- HessianOperator(Operator) to compute the hessian-vector product for a given pytorch model using subsampled data. Second, we upload data (i.e. the MNIST dataset), set training parameters and build pytorch model to be trained. After that the desired tridiagonal matrix T and matrix V with orthogonal rows are calculated via the lanczos function from lanczos_upd.py.
