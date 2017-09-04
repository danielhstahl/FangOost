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

/**
    This library is used to implement in a generic manner Fang Oosterlee (2007).
    The library is split into two sections: a "Levy" section and a standard CF
    section.  Recall that in Fang Oosterlee the CF is exp(ux_0+\phi).  The
    value of x_0 is actually log(S/K) and K iterates over all the strikes.
    Hence the extra exp(X) is needed for the Levy sections. 

    The non-levy sections are useful for standard inversion and expectation
    techniques using CFs.  The non-levy is more efficient than the levy 
    sections.

*/
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
    /*template<typename Number>
    auto halfFirstIndex(Number&& val){
        val[0]=.5*val[0];
        return std::move(val);
    }*/


    template<typename Number, typename Index>
    auto halfFirstIndexFn(const Number& val, const Index& index){
        return index==0?val*.5:val;
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
    auto formatCFReal(const U& u, const Number& xMin, const Number& cp, CF&& fnInv){
        return (fnInv(u)*exp(-u*xMin)).real()*cp;
    }

    template<typename U, typename Number, typename CF>
    auto formatCF(const U& u, const Number& xMin, const Number& cp, CF&& fnInv){
        return fnInv(u)*exp(-u*xMin)*cp;
    }
    
    /**
        Helper function to get complex u
        @u The real valued complex component.  Can be computed using getU(du, index)
    */
    template<typename Number>
    auto getComplexU(const Number& u){
        return std::complex<Number>(0, u);
    }

    /*Using X only makes sense for 
    Levy processes where log(S/K) changes 
    for every iteration.  This is done 
    separately from the Characteristic
    Function for computation purposes.*/
    template<typename Cmpl, typename X, typename Number, typename Index, typename VK>
    auto convoluteLevy(const Cmpl& cfIncr, const X& x, const Number& u, const Index& uIndex, VK&& vK){
        return (cfIncr*exp(getComplexU(u)*x)).real()*vK(u, x, uIndex);
    }
    //standard convolution in fouirer space (ie, multiplication)  */
    template<typename X, typename Number, typename Index, typename VK>
    auto convolute(const Number& cfIncr, const X& x, const Number& u, const Index& uIndex, VK&& vK){
        return cfIncr*vK(u, x, uIndex);
    }

    

    /**used when aggregating log cfs and then having to invert the results
        @xMin min of real plane
        @xMax max of real plane
        @logAndComplexCF vector of complex log values of a CF.  
        @returns actual CF for inversion

        Note that the exp(logAndComplex-u*xMin) is equivalent to 
        the computation done in formatCFReal but with vector instead
        of the function itself
    
    */
    template<typename Number, typename CF>
    auto convertLogCFToRealExp(const Number& xMin, const Number& xMax, CF&& logAndComplexCF){ 
        Number du=computeDU(xMin, xMax);
        auto cp=computeCP(du); 
        return futilities::for_each_parallel_copy(logAndComplexCF, 
            [&](const auto& val, const auto& uIndex){
                return exp(val-getComplexU(getU(du, uIndex))*xMin).real()*cp;
            }
        );
    }


    /**return vector of complex elements of cf. 
    This is ONLY needed where the CF depends on 
    a changing "x": like for option pricing where 
    x=log(S/K) and K iterates  */
    template<typename Number, typename Index,typename CF>
    auto computeDiscreteCF(const Number& xMin, const Number& xMax, const Index& uDiscrete, CF&& fnInv){
        auto du=computeDU(xMin, xMax);
        auto cp=computeCP(du); 
        return futilities::for_each_parallel(0, uDiscrete, [&](const auto& index){
            return formatCF(getComplexU(getU(du, index)), xMin, cp, fnInv);
        });
    }
    /**return vector of real elements of cf. 
    This will work for nearly every type 
    of inversion EXCEPT where the CF depends on 
    a changing "x": like for option pricing where 
    x=log(S/K) and K iterates  */
    template<typename Number, typename Index,typename CF>
    auto computeDiscreteCFReal(const Number& xMin, const Number& xMax, const Index& uDiscrete, CF&& fnInv){
        auto du=computeDU(xMin, xMax);
        auto cp=computeCP(du); 
        return futilities::for_each_parallel(0, uDiscrete, [&](const auto& index){
            return formatCFReal(getComplexU(getU(du, index)), xMin, cp, fnInv);
        });
    }


    
    
    /**
        Computes the convolution given the discretized characteristic function.  
        @xDiscrete Number of discrete points in density domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @discreteCF Discretized characteristic function.  This is vector of complex numbers.
        @vK Function (parameters u and x, and index)  
        @returns approximate convolution
    */
    template<typename Index, typename Number, typename CFArray, typename VK>
    auto computeConvolutionLevy(const Index& xDiscrete, const Number& xMin, const Number& xMax, CFArray&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        Number dx=computeDX(xDiscrete, xMin, xMax);
        Number du=computeDU(xMin, xMax);
        return futilities::for_each_parallel(0, xDiscrete, [&](const auto& xIndex){
            auto x=getX(xMin, dx, xIndex);
            return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
                return convoluteLevy(halfFirstIndexFn(cfIncr, uIndex), x, getU(du, uIndex), uIndex, vK);
                //return (cfIncr*exp(getComplexU(u)*x)).real()*vK(u, x);
            });
        });
    }


    template<typename Array, typename CFArray, typename VK>
    auto computeConvolutionVectorLevy(Array&& xValues, CFArray&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        auto du=computeDU(xValues.front(), xValues.back());
        return futilities::for_each_parallel(xValues, [&](const auto& x, const auto& xIndex){
            return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
                return convoluteLevy(halfFirstIndexFn(cfIncr, uIndex), x, getU(du, uIndex), uIndex, vK);
            });
        });
    }

    template<typename Number, typename X, typename CFArray, typename VK>
    auto computeConvolutionAtPointLevy(const X& xValue, const Number& xMin, const Number& xMax, CFArray&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        auto du=computeDU(xMin, xMax);
        return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
            return convoluteLevy(halfFirstIndexFn(cfIncr, uIndex), xValue, getU(du, uIndex), uIndex, vK);
        });
    }

    /**
        Computes the convolution given the discretized characteristic function.  
        @xDiscrete Number of discrete points in density domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @discreteCF Discretized characteristic function.  This is vector of complex numbers.
        @vK Function (parameters u and x, and index)  
        @returns approximate convolution
    */
    template<typename Index, typename Number, typename CFArray, typename VK>
    auto computeConvolution(const Index& xDiscrete, const Number& xMin, const Number& xMax, CFArray&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        Number dx=computeDX(xDiscrete, xMin, xMax);
        Number du=computeDU(xMin, xMax);
        return futilities::for_each_parallel(0, xDiscrete, [&](const auto& xIndex){
            auto x=getX(xMin, dx, xIndex);
            return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
                return convolute(halfFirstIndexFn(cfIncr, uIndex), x, getU(du, uIndex), uIndex, vK);
            });
        });
    }


    template<typename Array, typename CFArray, typename VK>
    auto computeConvolutionVector(Array&& xValues, CFArray&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        auto du=computeDU(xValues.front(), xValues.back());
        return futilities::for_each_parallel(xValues, [&](const auto& x, const auto& xIndex){
            return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
                return convolute(halfFirstIndexFn(cfIncr, uIndex), x, getU(du, uIndex), uIndex, vK);
            });
        });
    }

    template<typename X, typename Number, typename CFArray, typename VK>
    auto computeConvolutionAtPoint(const X& xValue, const Number& xMin, const Number& xMax, CFArray&& discreteCF, VK&& vK){ //vk as defined in fang oosterlee
        auto du=computeDU(xMin, xMax);
        return futilities::sum(discreteCF, [&](const auto& cfIncr, const auto& uIndex){
            return convolute(halfFirstIndexFn(cfIncr, uIndex), xValue, getU(du, uIndex), uIndex, vK);
        });
    }




/********
    FROM HERE ON are the functions that should be used by external programs





**/

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
    template<typename Index, typename Number, typename CF>
    auto computeInvDiscrete(const Index& xDiscrete, const Number& xMin, const Number& xMax, CF&& discreteCF){
        return computeConvolution(xDiscrete, xMin, xMax, discreteCF, [&](const auto& u, const auto& x, const auto& k){
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
    template<typename Index, typename Number, typename CF>
    auto computeInv(const Index& xDiscrete, const Index& uDiscrete,  const Number& xMin, const Number& xMax, CF&& fnInv){  
        return computeInvDiscrete(xDiscrete, xMin, xMax, computeDiscreteCFReal(xMin, xMax, uDiscrete, fnInv));
    }
        
     /**
        Computes the density given a log characteristic function at the discrete points xRange in xmin, xmax.  See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @convertLogCFToRealExp vector of log characteristic function of the density at discrete U 
        @returns approximate density
    */
    template<typename Index, typename Number, typename CF>
    auto computeInvDiscreteLog(const Index& xDiscrete, const Number& xMin, const Number& xMax, CF&& logFnInv){
        return computeInvDiscrete(xDiscrete, xMin, xMax, convertLogCFToRealExp(xMin, xMax, logFnInv));
    }

    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeExpectationLevy(
        const Index& xDiscrete, 
        const Index& uDiscrete,  
        const Number& xMin, 
        const Number& xMax, 
        CF&& fnInv, 
        VK&& vK
    ){  
        return computeConvolutionLevy(
            xDiscrete, xMin, xMax, 
            //halfFirstIndex(
                computeDiscreteCF(xMin, xMax, uDiscrete, fnInv),
            //), 
            vK
        );
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeExpectationLevyDiscrete(
        const Index& xDiscrete,   
        const Number& xMin, 
        const Number& xMax, 
        CF&& fnInv, 
        VK&& vK
    ){  
        return computeConvolutionLevy(
            xDiscrete, xMin, xMax, 
            //halfFirstIndex(
                fnInv,
            //), 
            vK
        );
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. Only used for non-Levy processes.  See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeExpectation(
        const Index& xDiscrete, 
        const Index& uDiscrete,  
        const Number& xMin, 
        const Number& xMax, 
        CF&& fnInv, 
        VK&& vK
    ){  
        return computeConvolution(
            xDiscrete, xMin, xMax, 
            //halfFirstIndex(
                computeDiscreteCFReal(xMin, xMax, uDiscrete, fnInv),
            //), 
            vK
        );
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. Only used for non-Levy processes.  See Fang Oosterlee (2007) for more information.
        @xDiscrete Number of discrete points in density domain
        @uDiscrete Number of discrete points in the complex domain
        @xmin Minimum number in the density domain
        @xmax Maximum number in the density domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Index, typename Number, typename CF, typename VK>
    auto computeExpectationDiscrete(
        const Index& xDiscrete, 
        const Number& xMin, 
        const Number& xMax, 
        CF&& fnInv, 
        VK&& vK
    ){  
        return computeConvolution(
            xDiscrete, xMin, xMax, 
           //halfFirstIndex(
                fnInv,
            //), 
            vK
        );
    }

    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
        @xValues x values to compute the function at.
        @uDiscrete Number of discrete points in the complex domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Array, typename Index,typename CF, typename VK>
    auto computeExpectationVectorLevy(
        Array&& xValues, 
        const Index& uDiscrete,  
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionVectorLevy(
            xValues, 
            //halfFirstIndex(
                computeDiscreteCF(
                    xValues.front(), 
                    xValues.back(), 
                    uDiscrete, fnInv
                ),
           // ), 
            vK
        );
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points xRange in xmin, xmax and functions of the expectation vK: E[f(vk)]. This is used if the CF is of a Levy process.  See Fang Oosterlee (2007) for more information.
        @xValues x values to compute the function at.
        @uDiscrete Number of discrete points in the complex domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Array,typename CF, typename VK>
    auto computeExpectationVectorLevyDiscrete(
        Array&& xValues, 
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionVectorLevy(
            xValues, 
           // halfFirstIndex(
                fnInv,
            //), 
            vK
        );
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points in xValues and functions of the expectation vK: E[f(vk)]. This is used if the CF is not for a Levy process.  See Fang Oosterlee (2007) for more information.
        @xValues x values to compute the function at.
        @uDiscrete Number of discrete points in the complex domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Array, typename Index,typename CF, typename VK>
    auto computeExpectationVector(
        Array&& xValues, 
        const Index& uDiscrete,  
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionVector(
            xValues, 
            //halfFirstIndex(
            computeDiscreteCFReal(
                xValues.front(), 
                xValues.back(), 
                uDiscrete, fnInv
            ),
            //), 
            vK
        );
    }
    /**
        Computes the expectation given a characteristic function fnInv at the discrete points in xValues and functions of the expectation vK: E[f(vk)]. This is used if the CF is not for a Levy process.  See Fang Oosterlee (2007) for more information.
        @xValues x values to compute the function at.
        @uDiscrete Number of discrete points in the complex domain
        @fnInv Characteristic function of the density.  May be computationally intense to compute (eg, if using the combined CF of millions of loans)
        @vK Function (parameters u and x) or vector to multiply discrete characteristic function by.  
        @returns approximate expectation
    */
    template<typename Array, typename CF, typename VK>
    auto computeExpectationVectorDiscrete(
        Array&& xValues, 
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionVector(
            xValues, 
            //halfFirstIndex(
                fnInv,
            //), 
            vK
        );
    }

    template<typename Number, typename X, typename Index,typename CF, typename VK>
    auto computeExpectationPointLevy(
        const X& xValue, 
        const Number& xMin, 
        const Number& xMax, 
        const Index& uDiscrete,  
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionAtPointLevy(
            xValue, 
            xMin,
            xMax,
            //halfFirstIndex(
                computeDiscreteCF(
                    xMin, 
                    xMax, 
                    uDiscrete, 
                    fnInv
                ),
            //), 
            vK
        );
    }
    template<typename Number, typename X, typename CF, typename VK>
    auto computeExpectationPointLevyDiscrete(
        const X& xValue, 
        const Number& xMin, 
        const Number& xMax, 
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionAtPointLevy(
            xValue, 
            xMin,
            xMax,
            //halfFirstIndex(
                fnInv,
            //), 
            vK
        );
    }
    template<typename X, typename Number, typename Index,typename CF, typename VK>
    auto computeExpectationPoint(
        const X& xValue, 
        const Number& xMin, 
        const Number& xMax, 
        const Index& uDiscrete,  
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionAtPoint(
            xValue, 
            xMin,
            xMax,
            //halfFirstIndex(
                computeDiscreteCFReal(
                    xMin, 
                    xMax, 
                    uDiscrete, 
                    fnInv
                ),
            //), 
            vK
        );
    }
    template<typename X, typename Number, typename CF, typename VK>
    auto computeExpectationPointDiscrete(
        const X& xValue, 
        const Number& xMin, 
        const Number& xMax, 
        CF&& fnInv, 
        VK&& vK
    ){
        return computeConvolutionAtPoint(
            xValue, 
            xMin,
            xMax,
            //halfFirstIndex(
                fnInv,
            //), 
            vK
        );
    }

    



}
#endif
