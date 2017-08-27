#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "FunctionalUtilities.h"
#include <iostream>
#include "FangOost.h"
#include <complex>
//u is k*pi/(b-a)
template<typename Number, typename Index>
auto VkCDF(const Number& x, const Number& u, const Number& a, const Number& b, const Index& k){
    //auto constant=(b-a)/(k*M_PI);
    return k==0?x-a:sin((x-a)*u)/u;
}
TEST_CASE("Test computeXRange", "[FangOost]"){
    REQUIRE(fangoost::computeXRange(5, 0.0, 1.0)==std::vector<double>({0, .25, .5, .75, 1.0})); 
}
TEST_CASE("Test computeInv", "[FangOost]"){
    const double mu=2;
    const double sigma=1;
    const int numX=5;
    const int numU=256;
    const double xMin=-3;
    const double xMax=7;
    auto normCF=[&](const auto& u){ //normal distribution's CF
        return exp(u*mu+.5*u*u*sigma*sigma);
    };      
    std::vector<double> referenceNormal=fangoost::computeXRange(numX, xMin, xMax);
    referenceNormal=futilities::for_each(std::move(referenceNormal), [&](double x, double index){ 
        return exp(-pow(x-mu, 2)/(2*sigma*sigma))/(sqrt(2*M_PI)*sigma);
    });
    auto myInverse=fangoost::computeInv(numX, numU, xMin, xMax, normCF);
    for(int i=0; i<numX; ++i){
        REQUIRE(myInverse[i]==Approx(referenceNormal[i]));
    }    
    
} 
TEST_CASE("Test computeInvDiscrete", "[FangOost]"){
    const double mu=2;
    const double sigma=1;
    const int numX=5;
    const int numU=256;
    const double xMin=-3;
    const double xMax=7;
    auto normCF=[&](const auto& u){ //normal distribution's CF
        return exp(u*mu+.5*u*u*sigma*sigma);
    };      
    std::vector<double> referenceNormal=fangoost::computeXRange(numX, xMin, xMax);
    referenceNormal=futilities::for_each(std::move(referenceNormal), [&](double x, double index){ 
        return exp(-pow(x-mu, 2)/(2*sigma*sigma))/(sqrt(2*M_PI)*sigma);
    });
    auto du=fangoost::computeDU(xMin, xMax);
    auto cp=fangoost::computeCP(du);
    auto discreteCF=futilities::for_each_parallel(0, numU, [&](const auto& index){
        return fangoost::formatCFReal(fangoost::getComplexU(fangoost::getU(du, index)), xMin, cp, normCF);
    });
    auto myInverse=fangoost::computeInvDiscrete(numX,  xMin, xMax, std::move(discreteCF));
    for(int i=0; i<numX; ++i){
        REQUIRE(myInverse[i]==Approx(referenceNormal[i]));
    }    
} 


TEST_CASE("Test computeInvDiscrete for two gaussian added", "[FangOost]"){
    const double mu=2;
    const double sigma=1;
    const double combinedMu=2*mu;
    const double combinedSig=sqrt(sigma*2);
    const int numX=5;
    const int numU=256;
    const double xMin=-4;
    const double xMax=12;
    auto normCF=[&](const auto& u){ //normal distribution's CF
        return u*mu+.5*u*u*sigma*sigma;
    };      
    std::vector<double> referenceNormal=fangoost::computeXRange(numX, xMin, xMax);
    std::vector<double> xRange=fangoost::computeXRange(numX, xMin, xMax);
    referenceNormal=futilities::for_each(std::move(referenceNormal), [&](double x, double index){ 
        return exp(-pow(x-combinedMu, 2)/(2*combinedSig*combinedSig))/(sqrt(2*M_PI)*combinedSig);
    });
    auto du=fangoost::computeDU(xMin, xMax);
    //
    auto discreteCF=futilities::for_each_parallel(0, numU, [&](const auto& index){
        return normCF(fangoost::getComplexU(fangoost::getU(du, index)));
    });
    for(auto& val:discreteCF){
        val=2.0*val;
    }
    auto myInverse=fangoost::computeInvDiscreteLog(numX,  xMin, xMax, std::move(discreteCF));
    for(int i=0; i<numX; ++i){
        REQUIRE(myInverse[i]==Approx(referenceNormal[i]));
    }   
} 

TEST_CASE("Test CDF", "[FangOost]"){
    const double mu=2;
    const double sigma=5;
    const int numX=55;
    const int numU=256;
    const double xMin=-20;
    const double xMax=25;
    auto normCF=[&](const auto& u){ //normal distribution's CF
        return exp(u*mu+.5*u*u*sigma*sigma);
    };      
    std::vector<double> referenceNormal=fangoost::computeXRange(numX, xMin, xMax);
    referenceNormal=futilities::for_each(std::move(referenceNormal), [&](double x, double index){ 
        return .5*erfc(-((x-mu)/sigma)/sqrt(2.0));
        //return exp(-pow(x-mu, 2)/(2*sigma*sigma))/(sqrt(2*M_PI)*sigma);
    });
    auto myCDF=fangoost::computeExpectation(numX, numU, xMin, xMax, normCF, [&](const auto& u, const auto& x, const auto& k){
        return VkCDF(x, u, xMin, xMax, k);
    });
    for(int i=0; i<numX; ++i){
        //std::cout<<myCDF[i]<<", "<<referenceNormal[i]<<std::endl;
        REQUIRE(myCDF[i]==Approx(referenceNormal[i]));
    }    
    
} 

TEST_CASE("Test computeExpectationVector", "FangOost"){
    int numX=100;
    int numU=100;
    double xmin=-5;
    double xmax=5;
    const double mu=2;
    const double sigma=1;
    auto normCF=[&](const auto& u){ //normal distribution's CF
        return u*mu+.5*u*u*sigma*sigma;
    }; 
    auto vk=[&](const auto& u, const auto& x, const auto& k){
        return cos(u*(x-xmin));
    };
    auto result1=fangoost::computeExpectation(numX, numU, xmin, xmax, normCF, vk);
    double dx=fangoost::computeDX(numX, xmin, xmax);
    auto xArray=futilities::for_each_parallel(0, numX, [&](const auto& index){
        return fangoost::getX(xmin, dx, index);
    });

    auto result2=fangoost::computeExpectationVector(xArray, numU, normCF, vk);
    for(int i=0; i<numX; ++i){
        REQUIRE(result1[i]==Approx(result2[i]));
    }
}