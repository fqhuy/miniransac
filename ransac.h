//
// Created by Phan Quoc Huy on 10/17/17.
//

#ifndef RANSAC_RANSAC_H
#define RANSAC_RANSAC_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <boost/shared_ptr.hpp>
#include <random>

using namespace boost::numeric::ublas;

#define EPS std::numeric_limits<double>::epsilon()

template <class T> class Model {
    /*
     * Virtual class for all shape models Line, Plane, Curve etc.
     */
public:
    Model(const boost::shared_ptr<const matrix<T> > &data ):
            _data(data), _n_samples(0), _n_params(0) {
    }

    // fit a set of samples and output the parameters
    virtual bool fit(const vector<int>& sample_ids, vector<T> &params) = 0;

    // prediction for new data points
    virtual void predict(const boost::shared_ptr<const matrix<T> > &X, vector<T> &y) = 0;

    virtual ~Model () {};
    inline unsigned int n_samples() {return _n_samples;}
    inline unsigned int n_params() {return _n_params;}
    inline boost::shared_ptr<const matrix<T> > data() {return _data;}
protected:
    boost::shared_ptr<const matrix<T> > _data;
    unsigned int _n_samples;
    unsigned int _n_params;
};

template <class T> class LineModel: public Model<T> {
public:
    /*
     * main constructor
     * @param data: matrix of data points, shape n_points x dimensionality
     */
    LineModel(const boost::shared_ptr<const matrix<T> > &data):
            Model<T>(data) {
        _n_params = 2;
        _n_samples = 2;
        _params = zero_vector<T>(2);
    }

    /*
     * optimise model params using a few sample points
     * @param sample_ids: sample indices
     * @param params: output parameters
     */
    bool fit(const vector<int>& sample_ids, vector<T> &params);
    void predict(const boost::shared_ptr<const matrix<T> > &X, vector<T> &y);

    ~LineModel() {}

    inline vector<T> const params() {
        return _params;
    }
private:
    using Model<T>::_n_params;
    using Model<T>::_n_samples;
    using Model<T>::_data;
    vector<T> _params;
};

template <class T> class Ransac {
public:
    /*
     * main constructor
     * @param model: one of the subclass of Model
     * @param thresh: inliner threshold
     * @param prob: confidence level
     * @param inliner_frac: fraction of inliners, Ransac will auto decide this value if it is set to negative
     * @param max_iters: maximum number of iterations
     */
    Ransac(const boost::shared_ptr<Model<T> > &model, const double thresh, const double prob, const double inliner_frac,
           const unsigned max_iters):
            _model(model), _prob(prob), _inliner_frac(inliner_frac),
            _max_iters(max_iters), _thresh(thresh)
    {
        std::random_device rd;
        _random_engine = std::mt19937_64(rd());
    }

    /* optimise model's paramters
     * @param inline_mask: true values indicate inliners
     * @param params: model's parameters, should be preallocated
     * @return : true if success, false if failed
     */
    bool fit(vector<bool> &inline_mask, vector<T> &params);

    // Compute the distances between groundtruth and predicted vectors
    inline void loss(const vector<T> &x, const vector<T> &x_pred, vector<T> &loss) {
        vector<T> tmp = x - x_pred;
        loss = element_prod(tmp, tmp);
        for(unsigned i=0; i < loss.size(); ++i) {
            loss(i) = sqrt(loss(i));
        }

    }

    // select random points from the data set
    inline void sample(const unsigned int n_samples, vector<int> &sample_ids) {
        vector<unsigned> indices = zero_vector<unsigned>(n_samples);
        for(unsigned i=0; i < n_samples;++i)
            indices(i) = i;

        std::shuffle(indices.begin(), indices.end(), _random_engine);
        for(unsigned i=0; i < sample_ids.size(); i++) {
            sample_ids(i) = indices(i);
        }
    }

    ~Ransac() {}
private:
    std::mt19937_64 _random_engine;
    boost::shared_ptr<Model<T> > _model;
    double _prob;
    unsigned int _max_iters;
    double _thresh;
    double _inliner_frac;
};

#endif //RANSAC_RANSAC_H
