/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2026 Yang Ye

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

#include <ql/experimental/inflation/jarrowyildirim.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <algorithm>
#include <cmath>
#include <utility>

namespace QuantLib {

    namespace {

        // int_0^t e^{-a s} ds
        Real expIntegral(Real a, Time t) { return -std::expm1(-a * t) / a; }

        // B(t,T) of a Hull-White rate with mean reversion a, for T - t = s
        Real bond(Real a, Time s) { return expIntegral(a, s); }

        // int_0^T B_x(u,T) du and the like, with s = T - u
        Real integralB(Real a, Time T) { return (T - expIntegral(a, T)) / a; }

        Real integralBB(Real a, Real b, Time T) {
            return (T - expIntegral(a, T) - expIntegral(b, T) + expIntegral(a + b, T)) /
                   (a * b);
        }

        // int_0^S e^{-a_r s} B_n(s) ds, with B_n(s) = (1 - e^{-a_n s})/a_n
        Real integralExpB(Real ar, Real an, Time S) {
            return (expIntegral(ar, S) - expIntegral(ar + an, S)) / an;
        }

        void checkCorrelations(Real rhoNR, Real rhoNI, Real rhoRI) {
            for (Real rho : {rhoNR, rhoNI, rhoRI})
                QL_REQUIRE(rho >= -1.0 && rho <= 1.0,
                           "correlation " << rho << " outside [-1, 1]");
            Real det = 1.0 - rhoNR * rhoNR - rhoNI * rhoNI - rhoRI * rhoRI +
                       2.0 * rhoNR * rhoNI * rhoRI;
            QL_REQUIRE(det >= -1.0e-12,
                       "correlation matrix is not positive semi-definite "
                       "(determinant " << det << ")");
        }

    }

    JarrowYildirim::JarrowYildirim(Handle<YieldTermStructure> nominalCurve,
                                   Handle<YieldTermStructure> realCurve,
                                   Real nominalMeanReversion,
                                   Real nominalVolatility,
                                   Real realMeanReversion,
                                   Real realVolatility,
                                   Real indexVolatility,
                                   Real rhoNominalReal,
                                   Real rhoNominalIndex,
                                   Real rhoRealIndex,
                                   Real baseIndex)
    : nominalCurve_(std::move(nominalCurve)), realCurve_(std::move(realCurve)),
      an_(nominalMeanReversion), sn_(nominalVolatility), ar_(realMeanReversion),
      sr_(realVolatility), sigmaI_(indexVolatility), rhoNR_(rhoNominalReal), rhoNI_(rhoNominalIndex),
      rhoRI_(rhoRealIndex), baseIndex_(baseIndex) {
        QL_REQUIRE(!nominalCurve_.empty(), "empty nominal curve");
        QL_REQUIRE(!realCurve_.empty(), "empty real curve");
        QL_REQUIRE(nominalMeanReversion > 0.0, "nominal mean reversion must be positive");
        QL_REQUIRE(realMeanReversion > 0.0, "real mean reversion must be positive");
        QL_REQUIRE(nominalVolatility >= 0.0, "negative nominal volatility");
        QL_REQUIRE(realVolatility >= 0.0, "negative real volatility");
        QL_REQUIRE(indexVolatility >= 0.0, "negative index volatility");
        QL_REQUIRE(baseIndex > 0.0, "base index must be positive");
        checkCorrelations(rhoNominalReal, rhoNominalIndex, rhoRealIndex);
        registerWith(nominalCurve_);
        registerWith(realCurve_);
    }

    Real JarrowYildirim::forwardIndex(Time T) const {
        QL_REQUIRE(T >= 0.0, "negative time " << T);
        return baseIndex_ * realCurve_->discount(T, true) / nominalCurve_->discount(T, true);
    }

    Real JarrowYildirim::zcInflationVariance(Time T) const {
        QL_REQUIRE(T >= 0.0, "negative time " << T);
        return sigmaI_ * sigmaI_ * T + sr_ * sr_ * integralBB(ar_, ar_, T) +
               sn_ * sn_ * integralBB(an_, an_, T) -
               2.0 * rhoRI_ * sigmaI_ * sr_ * integralB(ar_, T) +
               2.0 * rhoNI_ * sigmaI_ * sn_ * integralB(an_, T) -
               2.0 * rhoNR_ * sn_ * sr_ * integralBB(an_, ar_, T);
    }

    Real JarrowYildirim::zcInflationOption(Option::Type type, Real strike, Time T) const {
        QL_REQUIRE(T > 0.0, "non-positive time " << T);
        Real stdDev = std::sqrt(std::max(zcInflationVariance(T), 0.0));
        return nominalCurve_->discount(T, true) *
               blackFormula(type, strike, forwardIndex(T) / baseIndex_, stdDev);
    }

    Real JarrowYildirim::yoyConvexity(Time S, Time T) const {
        QL_REQUIRE(S >= 0.0 && S < T, "invalid times: S = " << S << ", T = " << T);
        Real exponent =
            sr_ * bond(ar_, T - S) *
            (rhoRI_ * sigmaI_ * expIntegral(ar_, S) +
             rhoNR_ * sn_ * integralExpB(ar_, an_, S) -
             sr_ * (expIntegral(ar_, S) - expIntegral(2.0 * ar_, S)) / ar_);
        return std::exp(exponent);
    }

    Real JarrowYildirim::yoyForward(Time S, Time T) const {
        return forwardIndex(T) / forwardIndex(S) * yoyConvexity(S, T);
    }

}
