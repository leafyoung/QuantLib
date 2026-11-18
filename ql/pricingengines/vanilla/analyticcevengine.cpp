/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2018 Klaus Spanderen

 This file is part of QuantLib, a free-software/open-source library
 for financial quantitative analysts and developers - http://quantlib.org/

 QuantLib is free software: you can redistribute it and/or modify it
 under the terms of the QuantLib license.  You should have received a
 copy of the license along with this program; if not, please email
 <quantlib-dev@lists.sf.net>. The license is also available online at
 <https://www.quantlib.org/license.shtml>.

 This program is distributed in the hope that it will be useful, but WITHOUT
 ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 FOR A PARTICULAR PURPOSE.  See the license for more details.
*/

/*! \file analyticcevengine.cpp */

#include <ql/exercise.hpp>
#include <ql/math/functional.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/vanilla/analyticcevengine.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <utility>

namespace QuantLib {

    CEVCalculator::CEVCalculator(Real f0, Real alpha, Real beta)
    : f0_(f0),
      alpha_(alpha),
      beta_(beta),
      delta_((1.0-2.0*beta)/(1.0-beta)),
      x0_(X(f0)) { }

    Real CEVCalculator::X(Real f) const {
        return std::pow(f, 2.0*(1.0-beta_))/squared(alpha_*(1.0-beta_));
    }

    Real CEVCalculator::value(
        Option::Type optionType, Real strike, Time t) const {

        // At beta=1, dF = alpha*F*dW is exactly geometric Brownian motion
        // with volatility alpha: the CEV closed form degenerates (delta_'s
        // 1/(1-beta) factor blows up, feeding the chi-squared distributions
        // below a divide-by-zero/overflowing degrees-of-freedom parameter,
        // which throws rather than returning a sensible price) exactly
        // where the true model is just Black. Dispatch to the Black formula
        // for beta at or extremely close to 1, where the CEV closed form is
        // both undefined/unstable and numerically indistinguishable from it
        // anyway.
        if (std::fabs(beta_ - 1.0) < 1.0e-4) {
            return blackFormula(optionType, strike, f0_, alpha_*std::sqrt(t), 1.0);
        }

        typedef boost::math::non_central_chi_squared_distribution<Real>
            nc_chi2;

        const Real kTilde = X(strike);

        if (optionType == Option::Call) {
            if (delta_ < 2.0) {
                return f0_ * (1.0 - boost::math::cdf(
                         nc_chi2(4.0-delta_, x0_/t), kTilde/t))
                     - strike * boost::math::cdf(
                         nc_chi2(2.0-delta_, kTilde/t), x0_/t);
            }
            else {
                const Real g =
                    boost::math::gamma_p(0.5*delta_-1.0,x0_/(2.0*t));

                return f0_ * (g - boost::math::cdf(
                         nc_chi2(delta_-2.0, kTilde/t), x0_/t))
                     - strike * boost::math::cdf(
                         nc_chi2(delta_, x0_/t), kTilde/t);
            }
        }
        else if (optionType == Option::Put) {
            if (delta_ < 2.0) {
                return - f0_ * boost::math::cdf(
                           nc_chi2(4.0-delta_, x0_/t), kTilde/t)
                       + strike * (1.0 - boost::math::cdf(
                           nc_chi2(2.0-delta_, kTilde/t), x0_/t));
            }
            else {
                return - f0_ * boost::math::cdf(
                           nc_chi2(delta_-2.0, kTilde/t), x0_/t)
                       + strike * (1.0 - boost::math::cdf(
                           nc_chi2(delta_, x0_/t), kTilde/t));
            }
        }
        else
            QL_FAIL("unknown option type");

    }

    AnalyticCEVEngine::AnalyticCEVEngine(Real f0,
                                         Real alpha,
                                         Real beta,
                                         Handle<YieldTermStructure> discountCurve)
    : calculator_(ext::make_shared<CEVCalculator>(f0, alpha, beta)),
      discountCurve_(std::move(discountCurve)) {
        registerWith(discountCurve_);
    }

    void AnalyticCEVEngine::calculate() const {

        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "not an European option");

        ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::dynamic_pointer_cast<StrikedTypePayoff>(arguments_.payoff);
        QL_REQUIRE(payoff, "non-striked payoff given");

        const Date exerciseDate = arguments_.exercise->lastDate();

        results_.value = calculator_->value(
                payoff->optionType(),
                payoff->strike(),
                discountCurve_->timeFromReference(exerciseDate))
            * discountCurve_->discount(exerciseDate);
    }

}
