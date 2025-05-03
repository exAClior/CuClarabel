# Implementation Notes

## Drawbacks
It reports out of memory error. Is it possible to automatically tune the warp size so that it can fit the GPU memory while working at a lower performance?

## `CUDA.CUSOLVER.gesvdjBatched`
When PSD variable is of size larger than $32 \times 32$ `gesvdjBatched` will fail.

### `gesvdj!`
`gesvd` means generalized singular value decomposition, the `j` means that it uses a Jacobi method, while `gesvd` will use a QR method. [Ref](https://docs.nvidia.com/cuda/cusolver/index.html?highlight=gesvdj#cusolverdn-t-gesvdj)
Jacobi method provided better parallelism. 
`V` parameter in `gesvdj!` says it will need to compute the right singular vector, while `N` probably says only singular values are needed.

It is limited to $32 \times 32$ because the Jacobi method is not well suited for large matrices. 

### `gesdva`
`a` stands for *approximation*. It is recommended to only be used when the problem is well conditioned. [Ref](https://docs.nvidia.com/cuda/cusolver/index.html?highlight=gesvda#cusolverdn-t-gesvdastridedbatched) Because the `A = USV^T`'s `U` will be bounded by the precision of `S`. Not sure why this is the case.

## References
- [Nvidia Forum](https://forums.developer.nvidia.com/t/the-origin-of-the-m-32-and-n-32-limitations-in-gesvdjbatched/266434)
- [Github Issue](https://github.com/jax-ml/jax/discussions/17609)