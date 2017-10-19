/*
Vicon Algorithm Software Engineering Test

In C++, implement the RANSAC algorithm to robustly estimate a line from noisy 2d points.

Tools: can use STL, boost, and C++ using a modern C++ compiler(up to C++ 11).
Input : a set of 2d points with unknown uncertainties, and any required algorithm parameters.
        A file containing a set of test points will be provided.
Outputs : the line parameterized as two points, and which input points are inliers.

Consider how the code might be organised to be easy to test, for others to read, and to improve later.
Use of the internet is encouraged to become familiar with details the algorithm and its variants;
 copying/pasting directly from the internet is unlikely to be helpful.
Comments can be useful to explain design/implementation choices; there will also be an opportunity to explain
 these verbally.
Any reasonable extensions to the basic RANSAC algorithm would be welcome but please note in comments
 where these are made, and provide any relevant citations.

We have provided a simple command-line harness, including code to read and write the output.
Please feel free to remove the boost command line parser if you do not have access to boost.
*/

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include "ransac.h"
//#include <random>

using namespace boost::numeric::ublas;

// simple point representation
using TPoint = std::array<double, 2>;

// function to read a comma-separated list of 2d points
bool ReadInputFile(std::string const& Filename, std::vector<TPoint> & o_rData)
{
    o_rData.clear();

    std::ifstream File;
    File.open(Filename);
    if (!File.is_open())
    {
        return false;
    }

    double x, y;
    char sep;
    while (File >> x >> sep >> y)
    {
        o_rData.push_back({ x, y });
    }

    return true;
}


int main(int argc, char** argv)
{

    // setup the input parameters
    // filename of input data
    std::string InputFilename;

    // filename of output data
    std::string OutputFilename("output.txt");

    // required probability that a result will be generated from inliers-only
    double Confidence = 0.99;

    // threshold on error metric
    double Threshold = 0.4;

    // approximate expected inlier fraction
    double ApproximateInlierFraction = -0.5; //0.5;


    // setup the command line arguments
    namespace po = boost::program_options;
    po::options_description OptionsDescription("Allowed options");
    OptionsDescription.add_options()
            ("help", "produce help message")
            ("filename", po::value<std::string>(&InputFilename), "filename for input point data")
            ("out,o", po::value<std::string>(&OutputFilename)->default_value(OutputFilename), "filename for output data")
            ("confidence,c", po::value<double>(&Confidence)->default_value(Confidence), "Confidence")
            ("threshold,t", po::value<double>(&Threshold)->default_value(Threshold), "Threshold")
            ("inliner,i", po::value<double>(&ApproximateInlierFraction)->default_value(ApproximateInlierFraction), "Approximate Inlier Fraction")
            ;

    po::positional_options_description PositionalOptionsDescription;
    PositionalOptionsDescription.add("filename", -1);

    po::variables_map VariablesMap;
    po::store(po::command_line_parser(argc, argv).
            options(OptionsDescription).positional(PositionalOptionsDescription).run(), VariablesMap);
    po::notify(VariablesMap);

    if (VariablesMap.count("help"))
    {
        std::cout << OptionsDescription << "\n";
        return 1;
    }

    std::vector<TPoint> Data;
    if (!ReadInputFile(InputFilename, Data)) {
        std::cerr << "Unable to read point data from file '" << InputFilename << "'" << std::endl;
        return 1;
    }

    // Output variables ...
    // std::vector<bool> Inliers;
    TPoint Point0;
    TPoint Point1;
    bool bSuccess(false);

    boost::shared_ptr<matrix<double> > X_ptr(new matrix<double>(Data.size(), 2));
    // -------------------------------------------
    vector<bool> inline_mask = scalar_vector<bool>(Data.size(), false) ;
    vector<double> params(2);

    for(int i=0;i < Data.size(); ++i){
        (*X_ptr)(i, 0) = Data[i][0];
        (*X_ptr)(i, 1) = Data[i][1];
    }

    boost::shared_ptr<LineModel<double> > model(new LineModel<double>(X_ptr));
    Ransac<double> ransac(model, Threshold, Confidence, ApproximateInlierFraction, 1000);
    bSuccess = ransac.fit(inline_mask, params);

    // selecting 2 points for plotting from slop-intercept parameters
    Point0[0] = 0.0;
    Point0[1] = params(0);
    Point1[0] = 1.;
    Point1[1] = params(0) + params(1);
    // -------------------------------------------


    // if successful, write the line parameterized as two points, and each input point along with its inlier status
    if (bSuccess)
    {
        std::ofstream Out(OutputFilename);
        Out << Point0[0] << " " << Point0[1] << "\n";
        Out << Point1[0] << " " << Point1[1] << "\n";
        assert(inline_mask.size() == Data.size());
        for (size_t Index = 0; Index != Data.size(); ++Index)
        {
            Out << Data[Index][0] << " " << Data[Index][1] << " " << inline_mask(Index) << "\n";
        }
        Out.close();
    }

    return 0;
}