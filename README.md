# optimization-methods

Convex and non-convex optimization methods from scratch.
Key idea and implementation for each algorithm and use cases.
Notes are from STAT 6020 taught by Professor Tianxi Li.

## Convex Problems

### First Order Methods

#### Gradient Descent

- 1st order method without constraints

#### Proximal Gradient Descent

- 1st order methods with constraints.
- Optimization function is the sum of two convex functions.
  - First is complicated but differentiable.
  - Second is simple but not necessarily differentiable (i.e. L1 norm).

#### Sub-gradient Method

- Applies to non-differentiable problems, with or without constraints.
- Ideal step size is 1/t.

### Second Order Methods

#### Newton's Method

- 2nd order method without constraints
- Solve a quadratic approximation to the function at each iteration.
- Backtracking line search to find acceptable step size.
- Locally quadratic convergence (linear while far from local optimal).
- Can extend to problems with equality constraints.

#### Log-Barrier Method

- 2nd order method with constraints.

#### Primal-Dual Interior Point Methods

### Other Methods

#### Coordinate Descent

- Can solve non-convex problems where the non-convex portion is additive in each component.
- Solve each individual (or block) parameter one at a time.

## Non-convex problems

### Majorization-minimization (MM) strategy

- Surrogate function lies above the optimization function and is tangent at one point.
- The key idea is to construct a surrogate function in a smart way. 

### Difference of Convex Functions (DC)

### Expectation-Maximization (EM) algorithm

### Monte-Carlo methods





