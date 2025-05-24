## Parallel Layer Normalization

This is a PyTorch implementation of the Parallel Layer Normalization (PLN) as described in the paper [2]. 

> Reference:
>
> [1] Ni Y, Guo Y, Jia J, et al. On the nonlinearity of layer normalization[J]. arXiv preprint arXiv:2406.01255, 2024.
>
> [2] Ni Y, Liu Y, Sun W, et al. Parallel layer normalization for universal approximation[J]. arXiv preprint arXiv:2505.13142, 2025.

The file `pln.py` implements the function of PLN. We introduce some of the parameters here. 

`num_features`

- usually denotes the width of the network, or the channel numbers in CNN. 

`num_per_group` 

- It means how many neurons in a group to perform layer normalization together. 

- The recommended `num_per_group` is `8`. 
- We require that `num_features % num_per_group == 0`. 

`centering`

- Set this `Ture` if using PLN, while `False` if using PLS (Parallel Layer Scaling). 

`affine` 

- It means whether to need affine parameters in PLN. We recommend to set this as `True`. 

`dim`

- The dimension of the input data. 
- `dim=4` for the image data---on the `c` dimension of `b*c*w*h` data. 
- `dim=3` for the sequential data---on the `d` dimension of `b*s*d` data. 
- `dim=2` for the vector data---on the `d` dimension of `b*d` data. 

The `example.py` provide the usage that PLN can be used as a layer module in `nn.Sequential()`. 