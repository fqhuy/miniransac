//
// Created by Phan Quoc Huy on 10/17/17.
//

#include "ransac.h"
#include <boost/numeric/ublas/lu.hpp>

template class LineModel<double>;
template bool LineModel<double>::fit(const vector<int>& sample_ids, vector<double> &params);
template void LineModel<double>::predict(const boost::shared_ptr<const matrix<double> > &X, vector<double> &y);

template  class Ransac<double>;
template bool Ransac<double>::fit(vector<bool> &inline_mask, vector<double> &params);


//template<class T>
//bool inv(const matrix<T> &input, matrix<T> &inverse) {
//    matrix<T> A(input);
//    permutation_matrix<int> pm(A.size1());
//    unsigned long res = lu_factorize(A, pm);
//    if (res != 0) return false;
//
//    inverse.assign(identity_matrix<T>(A.size1()));
//    lu_substitute(A, pm, inverse);
//
//    return true;
//}

template<class T>
bool inv2x2(const matrix<T> &input, matrix<T> &inverse) {
    inverse(0, 1) = -input(0, 1);
    inverse(1, 0) = -input(1, 0);
    inverse(0, 0) = input(1, 1);
    inverse(1, 1) = input(0, 0);
    inverse /= (input(0, 0) * input(1, 1) - input(0, 1) * input(1, 0));
    return true;
}

template<class T>
bool LineModel<T>::fit(const vector<int> &sample_ids, vector<T> &params) {
    unsigned long n_samples = sample_ids.size();
    matrix<T> X(*_data);
    if (sample_ids.size() == 2) {
        // slope
        double denom = (X(sample_ids(0), 0) - X(sample_ids(1), 0));
        if (denom < EPS) {
            params(1) = std::numeric_limits<double>::max();
        } else {
            params(1) = (X(sample_ids(0), 1) - X(sample_ids(1), 1)) / denom;
        }

        // intercept = y1 - slope * x1
        params(0) = X(sample_ids(0), 1) - params(1) * X(sample_ids(0), 0);
        _params(0) = params(0);
        _params(1) = params(1);
        return true;
    } else if (sample_ids.size() > 2) {
        // linear least squares
        matrix<T> x = scalar_matrix<T>(n_samples, 2, 1.0);
        matrix<T> y = zero_matrix<T>(n_samples, 1);
        for (unsigned i = 0; i < n_samples; ++i) {
            x(i, 1) = X(sample_ids(i), 0);
            y(i, 0) = X(sample_ids(i), 1);
        }

        matrix<T> xtx = prod(trans(x), x);
        matrix<T> xtxinv(xtx);
        inv2x2(xtx, xtxinv);

        matrix<T> xtxinvxt = prod(xtxinv, trans(x));
        matrix<T> re = prod(xtxinvxt, y);
        params(0) = re(0, 0);
        params(1) = re(1, 0);
        _params(0) = re(0, 0);
        _params(1) = re(1, 0);
        return true;
    } else {
        return false;
    }
}

template<class T>
void LineModel<T>::predict(const boost::shared_ptr<const matrix<T> >& X, vector<T> &y) {
    matrix<T> tmpX = scalar_matrix<T>(X->size1(), 2, 1);
    column(tmpX, 1) = column((*X), 0);
    y = prod(tmpX, _params);
};

template<class T>
bool Ransac<T>::fit(vector<bool> &inline_mask, vector<T> &params) {
    boost::shared_ptr<const matrix<T> > X = _model->data();
    unsigned long n_samples = X->size1();
    unsigned long n_inliners = 0;
    unsigned long n_max_inliners = 0;
    double K = 0;
    double log_prob = log(1. - _prob);
    double inliner_frac = _inliner_frac;
    vector<int> inliners = zero_vector<int>(n_samples);
    vector<int> best_inliners = zero_vector<int>(n_samples);
    vector<T> y = column((*X), 1);

    for (unsigned i = 0; i < _max_iters; ++i) {

        vector<int> sample_ids(_model->n_samples());
        sample(n_samples, sample_ids);

        _model->fit(sample_ids, params);

        // TODO: check valid model

        // compute losses
        vector<T> y_pred = zero_vector<T>(n_samples);
        _model->predict(X, y_pred);
        vector<T> losses = zero_vector<T>(n_samples);
        loss(y, y_pred, losses);

        // collect inliners
        n_inliners = 0;
        for (unsigned j = 0; j < n_samples; ++j) {
            if (losses(j) < _thresh) {
                inliners.insert_element(n_inliners, j);
                ++n_inliners;
            }
        }

        // update best model so far
        if (n_inliners > n_max_inliners) {
            std::cout << "iter: " << i << ", n_inliners: " << n_inliners << std::endl;
            n_max_inliners = n_inliners;
            best_inliners = inliners;
            if (_inliner_frac < 0 ) {
                inliner_frac = 1. / n_samples * n_max_inliners;
            }
            double p = 1. - pow(inliner_frac, (double) _model->n_samples());
            if (p < EPS)
                p = EPS;

            if (p == 1.)
                p = 1. - EPS;

            K = log_prob / log(p);

            if (i >= K) {
                break;
            }
        }

    }

    // re-estimate parameters using linear least squares
    best_inliners.resize(n_max_inliners);
    _model->fit(best_inliners, params);
    for(unsigned i=0; i < n_max_inliners; ++i){
        inline_mask(best_inliners(i)) = true;
    }
    return true;
}