#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include "catch.hpp"
#include "FunctionalUtilities.h"
#include <iostream>
#include "FangOost.h"
#include <complex>

TEST_CASE("Test computeXRange", "[FangOost]"){
    REQUIRE(fangoost::computeXRange(5, 0.0, 1.0)==std::vector<double>({0, .25, .5, .75, 1.0})); 
}
/*TEST_CASE("Test computeURange", "[FangOost]"){
    REQUIRE(fangoost::computeURange(5, 0.0, 1.0)==std::vector<double>({0, .25, .5, .75}));
}*/ 
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
        return fangoost::formatCF(fangoost::getComplexU(fangoost::getU(du, index)), xMin, cp, normCF);
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
    auto cp=fangoost::computeCP(du);
    auto discreteCF1=futilities::for_each_parallel(0, numU, [&](const auto& index){
        return normCF(fangoost::getComplexU(fangoost::getU(du, index)));
    });

    std::vector<double> discreteCF;
    for(int i=0; i<discreteCF1.size();++i){
        discreteCF.emplace_back(exp(2.0*discreteCF1[i]-xMin*fangoost::getComplexU(fangoost::getU(du, i))).real()*cp);
    }

    auto myInverse=fangoost::computeInvDiscrete(numX,  xMin, xMax, std::move(discreteCF));
    for(int i=0; i<numX; ++i){
        //std::cout<<myInverse[i]<<", "<<referenceNormal[i]<<", "<<xRange[i]<<std::endl;
        REQUIRE(myInverse[i]==Approx(referenceNormal[i]));
    }    
} 