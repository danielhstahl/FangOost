#ifndef FANGOOST
#define FANGOOST
#include <complex>
#include <cmath>
#include "FunctionalUtilities.h"

namespace fangoost{
    /**
        Function to compute the discrete X range per Fang Oosterlee (2007)
        @xDiscrete number of sections to parse the X domain into
        @xMin the minimum of the X domain
        @xMax the maximum of the X domain
        @return vector of discrete X values
    */
    template<typename Number>
    std::vector<Number> computeXRange(int xDiscrete, const Number& xMin, const Number& xMax){
        return futilities::for_emplace_back(xMin, xMax, xDiscrete, [](const auto& val){
            return val;
        });
    }
    /**
        Function to compute the discrete U range per Fang Oosterlee (2007)
        @uDiscrete number of sections to parse the U domain into
        @xMin the minimum of the X domain
        @xMax the maximum of the X domain
        @return vector of discrete U values
    */
    template<typename Number>
    std::vector<Number> computeURange(int uDiscrete, const Number& xMin, const Number& xMax){
        Number uMax=(M_PI/(xMax-xMin))*(uDiscrete-1.0);
        return futilities::for_emplace_back(0, uMax, uDiscrete, [](const auto& val){
            return val;
        });
    }
    /**
        Function to compute the discrete X.  The operation is cheap and takes less ram than simply using the computeXRange function to create a vector
        @xMin the minimum of the X domain
        @dx the difference between the nodes in X
        @index the location of the node
        @return location of discrete X
    */
    auto getX(const auto& xMin, const auto& dx, const auto& index){
        return xMin+index*dx;
    }
    /**
        Function to compute the discrete U.  The operation is cheap and takes less ram than simply using the computeURange function to create a vector.  Note that "uMin" is always zero and hence is unecessary.  This can (should?) be simply an implementation of a generic "getNode" function but is broken into two functions to make it explicit and be more closely aligned with the Fang Oosterlee paper.
        @du the difference between the nodes in U
        @index the location of the node
        @return location of discrete U
    */
    auto getU(const auto& du, const auto& index){
        return index*du;
    }
    /**
        Function to compute the difference in successive X nodes.  This can feed into the "getX" function.
        @xDiscrete number of sections to parse the X domain into
        @xMin the minimum of the X domain
        @xMax the maximum of the X domain
        @return the difference between successive x nodes
    */
    auto computeDX(int xDiscrete, const auto& xMin,const auto& xMax){
        return (xMax-xMin)/(double)(xDiscrete-1);
    }
    /**
        Function to compute the difference in successive U nodes.  This can feed into the "getU" function.  Note that this depends on X: the U and X domains are not "independent".
        @xMin the minimum of the X domain
        @xMax the maximum of the X domain
        @return the difference between successive U nodes
    */
    auto computeDU(const auto& xMin,const auto& xMax){
        return M_PI/(xMax-xMin);
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Number>
    auto computeInv(int xDiscrete, int uDiscrete,  const Number& xMin, const Number& xMax, auto&& fnInv, auto&& vK){
        auto du=computeDU(xMin, xMax);
        auto cp=(2.0*du)/M_PI;
        auto halfFirstIndex=[](auto&& val){
            val[0]=.5*val[0];
            return std::move(val);
        };
        return computeConvolution(xDiscrete, uDiscrete, xMin, xMax, halfFirstIndex(futilities::for_each_parallel(0, uDiscrete, [&](const auto& index){
            auto val=getU(du, index);
            auto u=std::complex<Number>(0, val);
            return (fnInv(u)*exp(-u*xMin)).real()*cp;
        })), vK);
    }
    /**
        Computes a discrete density corresponding to the characteristic function fnInv at the discrete points xRange in xmin, xmax.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density
        @returns approximate density
    */
    template<typename Number>
    auto computeInv(int xDiscrete, int uDiscrete,  const Number& xMin, const Number& xMax, auto&& fnInv){
        return computeInv(xDiscrete, uDiscrete,  xMin, xMax, fnInv, [&](const auto& u, const auto& x){
            return cos(u*(x-xMin));
        });
    }
    /**
        Computes the convolution given the discretized characteristic function.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Discretized characteristic function
        @returns approximate convolution
    */
    template<typename Number>
    std::vector<Number> computeConvolution(int xDiscrete, int uDiscrete, const Number& xMin, const Number& xMax, const auto& discreteCF, auto&& vK){ //vk as defined in fang oosterlee
        Number dx=computeDX(xDiscrete, xMin, xMax);
        Number du=computeDU(xMin, xMax);
        return futilities::for_each_parallel(0, xDiscrete, [&](const auto& xIndex){
            return futilities::sum(0, uDiscrete, [&](const auto& uIndex){
                return discreteCF[uIndex]*vK(getU(du, uIndex), getX(xMin, dx, xIndex));
            });
        });
    }
}
#endif