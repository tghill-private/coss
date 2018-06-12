# Parallel Programming Clusters with MPI
*June 12, 2018. University of Toronto, Wilson Hall. Ramses van Zon*.


## Scientific MPI Example
Intuitively, there are many different types of scientific problems that can inherently take advantage of MPI. Often this involves discretizing the continuous equations of the model on a grid.

### Discretizing Derivatives
Use a second-order centered difference method to discretize the diffusion equation

![DiffusionEq](https://latex.codecogs.com/gif.latex?\frac{\partial&space;T}{\partial&space;t}&space;=&space;K&space;\frac{\partial^2&space;T}{\partial&space;x^2})

And

![DiscretizeDeriv](https://latex.codecogs.com/gif.latex?\frac{\partial^2&space;T}{\partial&space;x^2}&space;\approx&space;\frac{T_{i&plus;1}&space;-&space;2T_{i}&space;&plus;&space;T_{i-1}}{\Delta&space;x^2})

This is written in 1-D, but we could discretize the equation in higher dimensions.

What about boundaries? We use *Guard Cells*. Pad the domain with guard cells so the stencil works for the first and last point in the domain. We fill the guard cells with values such that the boundary conditions are met.
