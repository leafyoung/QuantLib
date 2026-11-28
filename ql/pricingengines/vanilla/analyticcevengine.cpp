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
#include <ql/instruments/payoffs.hpp>
#include <ql/math/functional.hpp>
#include <ql/math/integrals/gaussianquadratures.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/vanilla/analyticcevengine.hpp>
#include <boost/math/distributions/non_central_chi_squared.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <algorithm>
#include <cmath>
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
        else {
            QL_FAIL("unknown option type");
}

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


    Time cevEffectiveTime(
        Real alpha, Real beta,
        const ext::shared_ptr<YieldTermStructure>& discountCurve,
        Time start, Time end) {

        QL_REQUIRE(discountCurve, "no discount curve given");
        QL_REQUIRE(start < end,
                   "cevEffectiveTime requires a positive time interval");

        const DiscountFactor dStart = discountCurve->discount(start);
        const Real exponent = 2.0*(1.0-beta);

        // the integrand is smooth and positive for flat rate curves and
        // piecewise smooth otherwise; one 64 point Gauss-Legendre rule per
        // year of the interval resolves it to about 3e-5 relative at a curve
        // kink for forward rate jumps up to 20%, and to about 1e-4 for a
        // jump of 100%. The rule is built once: the FDM boundary evaluates
        // this at every time step.
        static const GaussLegendreIntegration quadrature(64);
        const Size panels = std::max<Size>(1, Size(std::ceil(end-start)));
        const Time halfLength = 0.5*(end-start)/Real(panels);

        Real integral = 0.0;
        for (Size k = 0; k < panels; ++k) {
            const Time mid = start + (2.0*Real(k)+1.0)*halfLength;
            integral += halfLength*quadrature(
                [&](Real x) -> Real {
                    const Time s = mid + halfLength*x;
                    return std::pow(discountCurve->discount(s)/dStart,
                                    exponent);
                });
        }
        return squared(alpha)*integral;
    }

    namespace {
        // Black-Scholes value at the local lognormal volatility f0^(beta-1)
        // of the unit-volatility process run for the effective time
        Real cevBlackLimitValue(Option::Type type, Real strike,
                                const CEVCalculator& calculator,
                                const ext::shared_ptr<YieldTermStructure>& curve,
                                Time maturity, DiscountFactor discount) {
            const Real effectiveTime = cevEffectiveTime(
                calculator.alpha(), calculator.beta(), curve, 0.0, maturity);
            return blackFormula(
                type, strike, calculator.f0()/discount,
                std::pow(calculator.f0(), calculator.beta()-1.0)
                    *std::sqrt(effectiveTime),
                discount);
        }
    }

    AnalyticDriftedCEVEngine::AnalyticDriftedCEVEngine(Real f0,
                                                       Real alpha,
                                                       Real beta,
                                                       Handle<YieldTermStructure> discountCurve)
    : calculator_(ext::make_shared<CEVCalculator>(f0, alpha, beta)),
      discountCurve_(std::move(discountCurve)) {
        registerWith(discountCurve_);
    }

    void AnalyticDriftedCEVEngine::calculate() const {

        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "not an European option");

        ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::dynamic_pointer_cast<PlainVanillaPayoff>(arguments_.payoff);
        QL_REQUIRE(payoff, "only plain vanilla payoffs are supported");

        const Date exerciseDate = arguments_.exercise->lastDate();
        const Time maturity = discountCurve_->timeFromReference(exerciseDate);

        QL_REQUIRE(maturity > 0.0,
                   "the option matures at or before the discount curve's reference date");

        const DiscountFactor discount = discountCurve_->discount(exerciseDate);
        const Real beta = calculator_->beta();

        if (std::fabs(beta - 1.0) < cevBlackLimitBandWidth) {
            // the CEV price converges to its Black-Scholes limit as beta
            // approaches one. Within the band the closed form degenerates
            // towards infinite degrees of freedom and boost's noncentral
            // chi-squared evaluation can raise, while the deviation from the
            // limit stays within the model curvature of the neighbourhood
            results_.value = cevBlackLimitValue(
                payoff->optionType(), payoff->strike(), *calculator_,
                discountCurve_.currentLink(), maturity, discount);
            return;
        }

        // Remove the deterministic drift: the option value is the
        // undiscounted value of the drift-removed unit volatility CEV
        // process at the effective time, at the discounted strike.
        // Exact for deterministic rate curves up to the quadrature accuracy
        // of cevEffectiveTime.
        const CEVCalculator unitVolCalculator(calculator_->f0(), 1.0, beta);

        results_.value = unitVolCalculator.value(
            payoff->optionType(),
            payoff->strike()*discount,
            cevEffectiveTime(calculator_->alpha(), beta,
                             discountCurve_.currentLink(), 0.0, maturity));
    }

}
