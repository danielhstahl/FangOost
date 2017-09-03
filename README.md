
| [Linux][lin-link] | [Windows][win-link] | [Codecov][cov-link] |
| :---------------: | :-----------------: | :-------------------: |
| ![lin-badge]      | ![win-badge]        | ![cov-badge]          |

[lin-badge]: https://travis-ci.org/phillyfan1138/FangOost.svg?branch=master "Travis build status"
[lin-link]:  https://travis-ci.org/phillyfan1138/FangOost "Travis build status"
[win-badge]: https://ci.appveyor.com/api/projects/status/767nlo0xuw4pinj8?svg=true "AppVeyor build status"
[win-link]:  https://ci.appveyor.com/project/phillyfan1138/fangoost "AppVeyor build status"
[cov-badge]: https://codecov.io/gh/phillyfan1138/FangOost/branch/master/graph/badge.svg
[cov-link]:  https://codecov.io/gh/phillyfan1138/FangOost

This is a generic implementation of the [Fang-Oosterlee](http://ta.twi.tudelft.nl/mf/users/oosterle/oosterlee/COS.pdf) paper for inverting density functions and convolutions.  Depends on my [Functional Utilities](https://github.com/phillyfan1138/FunctionalUtilities) library.

The library falls into two categories: a "Levy" section which assumes that the characteristic function and the integrand depend on a common variable, and a standard section where the characteristic function and integrand do not depend on a common variable.  To price options, the characteristic function is of the form exp(ui x_0+\phi(u)) where x_0 is equal to log(S/K) and K iterates over all required strikes.  Hence when integrating, the characteristic function must be re-evaluated at every interval.  Without this requirement, the algorithm is more efficient since the characteristic function can be evaluated once for every discrete u.  

To see how this algorithm can be used, see the [test](./test.cpp) code, my [FFTOptionPricing](https://github.com/phillyfan1138/FFTOptionPricing) repo, and my [cfdistutilities](https://github.com/phillyfan1138/cfdistutilities) repo.

## Some API oddities

The `computeInvDiscrete` functions (and some of the other discrete functions) assume that the CF has already been multiplied by `cp`; see the test cases with `computeInvDiscrete`.  However, some discrete functions do take a "raw" CF: see for example `computeInvDiscreteLog`.  The reason is that adding `cp` is not added to the `computeInvDiscrete` function is for efficiency purposes.  

## Limitations

For densities without derivatives of all orders, the convergence may be slow. For example, Beta distributions may not converge at all when the mode of the distribution is near zero or one.