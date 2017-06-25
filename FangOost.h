#ifndef FANGOOST
#define FANGOOST
#include <complex>
#define _USE_MATH_DEFINES
#include <cmath>
#include <type_traits>
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#include "FunctionalUtilities.h"

/*#ifndef __IS_COMPLEX
#define __IS_COMPLEX
template<typename T> struct is_complex_vector: public std::false_type {};
template<typename T, typename A> struct is_complex_vector<std::vector<T, A>> : public std::true_type {};
#endif*/
namespace fangoost{
    /**
        Function to compute the discrete X range per Fang Oosterlee (2007)
        @xDiscrete number of sections to parse the X domain into
        @xMin the minimum of the X domain
        @xMax the maximum of the X domain
        @return vector of discrete X values
    */
    template<typename Index, typename Number>
    auto computeXRange(const Index& xDiscrete, const Number& xMin, const Number& xMax){
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
    /*template<typename Index, typename Number>
    auto computeURange(const Index& uDiscrete, const Number& xMin, const Number& xMax){
        //Number uMax=(M_PI/(xMax-xMin))*(uDiscrete-1.0);
        return futilities::for_emplace_back(0, (M_PI/(xMax-xMin))*(uDiscrete-1.0), uDiscrete, [](const auto& val){
            return val;
        });
    }*/
    /**
        Function to compute the discrete X.  The operation is cheap and takes less ram than simply using the computeXRange function to create a vector
        @xMin the minimum of the X domain
        @dx the difference between the nodes in X
        @index the location of the node
        @return location of discrete X
    */
    template<typename Index, typename Number>
    auto getX(const Number& xMin, const Number& dx, const Index& index){
        return xMin+index*dx;
    }
    /**
        Function to compute the discrete U.  The operation is cheap and takes less ram than simply using the computeURange function to create a vector.  Note that "uMin" is always zero and hence is unecessary.  This can (should?) be simply an implementation of a generic "getNode" function but is broken into two functions to make it explicit and be more closely aligned with the Fang Oosterlee paper.
        @du the difference between the nodes in U
        @index the location of the node
        @return location of discrete U
    */
    template<typename Index, typename Number>
    auto getU(const Number& du, const Index& index){
        return index*du;
    }
    /**
        Function to compute the difference in successive X nodes.  This can feed into the "getX" function.
        @xDiscrete number of sections to parse the X domain into
        @xMin the minimum of the X domain
        @xMax the maximum of the X domain
        @return the difference between successive x nodes
    */
    template<typename Index, typename Number>
    auto computeDX(const Index& xDiscrete, const Number& xMin,const Number& xMax){
        return (xMax-xMin)/(double)(xDiscrete-1);
    }
    /**
        Function to compute the difference in successive U nodes.  This can feed into the "getU" function.  Note that this depends on X: the U and X domains are not "independent".
        @xMin the minimum of the X domain
        @xMax the maximum of the X domain
        @return the difference between successive U nodes
    */
    template<typename Number>
    auto computeDU(const Number& xMin,const Number& xMax){
        return M_PI/(xMax-xMin);
    }
    /**
        Helper function to make the first index half
        @val Container (eg std::vector) to apply to first index of
    */
    template<typename Number>
    auto halfFirstIndex(Number&& val){
        val[0]=.5*val[0];
        return std::move(val);
    }
    /**
        Helper function to get "CP"
        @du Discrete step in u.  Can be computed using computeDU(xMin, xMax)
    */
    template<typename Number>
    auto computeCP(const Number& du){
        return (2.0*du)/M_PI;
    }
    /**
        Helps convert CF into appropriate vector for inverting
        @u Complex number.  Discretiziation of complex plane.  Can be computed by calling getU(du, index)
        @xMin Minimum of real plane
        @cp Size of integration in complex domain.  Can be computed by calling computeCP(du)
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
    */
    template<typename U, typename Number, typename CF>
    auto formatCF(const U& u, const Number& xMin, const Number& cp, CF&& fnInv){
        return (fnInv(u)*exp(-u*xMin)).real()*cp;
    }
    
    /**
        Helper function to get complex u
        @u The real valued complex component.  Can be computed using getU(du, index)
    */
    template<typename Number>
    auto getComplexU(const Number& u){
        return std::complex<Number>(0, u);
    }
    /**
        Computes the convolution given the discretized characteristic function.
        @xDiscrete Number of discrete points in density domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @discreteCF Discretized characteristic function
        @vK Function (parameters u and x)  
        @returns approximate convolution
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeConvolution(const Index& xDiscrete, const Number& xMin, const Number& xMax, CF&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        Number dx=computeDX(xDiscrete, xMin, xMax);
        Number du=computeDU(xMin, xMax);
        return futilities::for_each_parallel(0, xDiscrete, [&](const auto& xIndex){
            auto x=getX(xMin, dx, xIndex);
            return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
                auto u=getU(du, uIndex);
                /**I dont require a exp(uC*x) like in computeConvolutionComplex because its already incorporated in the "VK", eg cos(u(x-xmin))*/
                return cfIncr*vK(u, x);
            });
        });
    }
    
    
    /**
        Computes the convolution given the discretized characteristic function.  More general than computeConvolution since it takes the discrete X values as well
        @xDiscrete Number of discrete points in density domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @discreteCF Discretized characteristic function
        @vK Function (parameters u and x)  
        @returns approximate convolution
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeConvolutionComplex(const Index& xDiscrete, const Number& xMin, const Number& xMax, CF&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        Number dx=computeDX(xDiscrete, xMin, xMax);
        Number du=computeDU(xMin, xMax);
        return futilities::for_each_parallel(0, xDiscrete, [&](const auto& xIndex){
            auto x=getX(xMin, dx, xIndex);
            return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
                auto u=getU(du, uIndex);
                auto uC=getComplexU(u);
                return (cfIncr*exp(uC*x)).real()*vK(u, x);
            });
        });
    }

    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeInv(const Index& xDiscrete, const Index& uDiscrete,  const Number& xMin, const Number& xMax, CF&& fnInv, VK&& vK){
        auto du=computeDU(xMin, xMax);
        auto cp=computeCP(du);    
        return computeConvolution(xDiscrete, xMin, xMax, halfFirstIndex(futilities::for_each_parallel(0, uDiscrete, [&](const auto& index){
            return formatCF(getComplexU(getU(du, index)), xMin, cp, fnInv);
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
    template<typename Index, typename Number, typename CF>
    auto computeInv(const Index& xDiscrete, const Index& uDiscrete,  const Number& xMin, const Number& xMax, CF&& fnInv){
        return computeInv(xDiscrete, uDiscrete,  xMin, xMax, fnInv, [&](const auto& u, const auto& x){
            return cos(u*(x-xMin));
        });
    }
     /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv vector of characteristic function of the density at discrete U
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeInvDiscrete(const Index& xDiscrete, const Number& xMin, const Number& xMax, CF&& fnInv, VK&& vK){
        return computeConvolution(xDiscrete, xMin, xMax, halfFirstIndex(fnInv), vK);
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv vector of characteristic function of the density at discrete U
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF>
    auto computeInvDiscrete(const Index& xDiscrete, const Number& xMin, const Number& xMax, CF&& fnInv){
        return computeInvDiscrete(xDiscrete, xMin, xMax, fnInv, [&](const auto& u, const auto& x){
            return cos(u*(x-xMin));
        });
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeExpectation(const Index& xDiscrete, const Index& uDiscrete,  const Number& xMin, const Number& xMax, CF&& fnInv, VK&& vK){
        auto du=computeDU(xMin, xMax);
        auto cp=computeCP(du);   
        return computeConvolutionComplex(xDiscrete, xMin, xMax, halfFirstIndex(futilities::for_each_parallel(0, uDiscrete, [&](const auto& index){
            return formatCF(getComplexU(getU(du, index)), xMin, cp, fnInv);
        })), vK);
    }
}
#endif
