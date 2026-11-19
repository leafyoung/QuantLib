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

#include "toplevelfixture.hpp"
#include "utilities.hpp"
#include <ql/experimental/inflation/jarrowyildirim.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/integrals/gausslobattointegral.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/matrixutilities/pseudosqrt.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/models/shortrate/onefactormodels/hullwhite.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>
#include <vector>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(JarrowYildirimTests)

namespace {

    struct Setup {
        Date today = Settings::instance().evaluationDate();
        ext::shared_ptr<SimpleQuote> nominalRate = ext::make_shared<SimpleQuote>(0.03);
        ext::shared_ptr<SimpleQuote> realRate = ext::make_shared<SimpleQuote>(0.012);
        Handle<YieldTermStructure> nominalCurve, realCurve;
        Real an = 0.08, sn = 0.015, ar = 0.05, sr = 0.012, sI = 0.08;
        Real rhoNR = 0.4, rhoNI = 0.2, rhoRI = -0.3;

        Setup() {
            nominalCurve = Handle<YieldTermStructure>(
                ext::make_shared<FlatForward>(today, Handle<Quote>(nominalRate), Actual365Fixed()));
            realCurve = Handle<YieldTermStructure>(
                ext::make_shared<FlatForward>(today, Handle<Quote>(realRate), Actual365Fixed()));
        }

        JarrowYildirim model() const {
            return JarrowYildirim(nominalCurve, realCurve, an, sn, ar, sr, sI, rhoNR, rhoNI, rhoRI);
        }
    };

    Real bFun(Real a, Real s) {
        return -std::expm1(-a * s) / a;
    }

    Real integrate(const std::function<Real(Real)>& f, Real a, Real b) {
        return GaussLobattoIntegral(100000, 1.0e-15)(f, a, b);
    }

}

BOOST_AUTO_TEST_CASE(testZeroVolatilityRecoversCurves) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim model with zero volatilities...");
    Setup s;
    s.sn = s.sr = s.sI = 0.0;
    JarrowYildirim jy = s.model();
    Real S = 2.0, T = 5.0;
    Real fS = std::exp((0.03 - 0.012) * S), fT = std::exp((0.03 - 0.012) * T);
    QL_CHECK_CLOSE(jy.forwardIndexRatio(T), fT, 1.0e-10);
    if (jy.yoyConvexity(S, T) != 1.0)
        BOOST_ERROR("convexity " << jy.yoyConvexity(S, T) << " expected exactly 1");
    Real expectedYoY = fT / fS;
    QL_CHECK_CLOSE(jy.yoyForward(S, T), expectedYoY, 1.0e-10);
    BOOST_CHECK_EQUAL(jy.zcInflationVariance(T), 0.0);
    Real df = std::exp(-0.03 * T);
    for (Real k : {1.05, 1.10, 1.20}) {
        Real call = jy.zcInflationOption(Option::Call, k, T);
        Real put = jy.zcInflationOption(Option::Put, k, T);
        Real expCall = df * std::max<Real>(fT - k, 0.0);
        Real expPut = df * std::max<Real>(k - fT, 0.0);
        QL_CHECK_CLOSE(call, expCall, 1.0e-10);
        QL_CHECK_CLOSE(put, expPut, 1.0e-10);
    }
}

BOOST_AUTO_TEST_CASE(testClosedFormsAgainstQuadrature) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim closed forms against quadrature...");
    // mean reversions from the small-a branch through the closed-form threshold
    // including one tiny and one large mean reversion, which exercise the
    // choice of integration panels and of the branch for each term
    for (auto rates :
         {std::make_pair(0.08, 0.05), std::make_pair(1.0e-8, 1.0e-6), std::make_pair(1.0e-3, 0.5),
          std::make_pair(0.02, 0.02), std::make_pair(0.15, 1.0e-7), std::make_pair(1.0e-3, 2.0),
          std::make_pair(1.0e-8, 1.0), std::make_pair(1.0e-8, 0.05), std::make_pair(2.0e-4, 2.0),
          std::make_pair(2.0, 1.0e-8), std::make_pair(1.0, 2.0e-4)}) {
        Setup s;
        s.an = rates.first;
        s.ar = rates.second;
        JarrowYildirim jy = s.model();
        for (Real T : {0.5, 3.0, 10.0, 30.0}) {
            auto integrand = [&](Real u) -> Real {
                Real bn = bFun(s.an, T - u), br = bFun(s.ar, T - u);
                return s.sI * s.sI + s.sr * s.sr * br * br + s.sn * s.sn * bn * bn -
                       2.0 * s.rhoRI * s.sI * s.sr * br + 2.0 * s.rhoNI * s.sI * s.sn * bn -
                       2.0 * s.rhoNR * s.sn * s.sr * bn * br;
            };
            Real expected = integrate(integrand, 0.0, T);
            Real calculated = jy.zcInflationVariance(T);
            QL_CHECK_CLOSE(calculated, expected, 1.0e-9);
        }
        for (auto st :
             {std::make_pair(1.0, 2.0), std::make_pair(4.0, 6.0), std::make_pair(0.0, 3.0),
              std::make_pair(7.0, 7.5), std::make_pair(10.0, 30.0), std::make_pair(30.0, 31.0)}) {
            Real S = st.first, T = st.second;
            auto integrand = [&](Real u) -> Real {
                Real delta = bFun(s.ar, T - u) - bFun(s.ar, S - u);
                return s.sr * delta *
                       (s.rhoRI * s.sI + s.rhoNR * s.sn * bFun(s.an, S - u) -
                        s.sr * bFun(s.ar, S - u));
            };
            Real expected = S > 0.0 ? integrate(integrand, 0.0, S) : Real(0.0);
            Real calculated = std::log(jy.yoyConvexity(S, T));
            QL_CHECK_CLOSE(calculated, expected, 1.0e-9);
        }
        BOOST_CHECK_EQUAL(jy.yoyConvexity(0.0, 3.0), 1.0);
    }
}

BOOST_AUTO_TEST_CASE(testZcCapletFloorletParity) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim zero-coupon caplet-floorlet parity...");
    Setup s;
    JarrowYildirim jy = s.model();
    for (Real T : {1.0, 5.0, 10.0}) {
        for (Real k : {0.95, 1.0, 1.1, 1.3}) {
            Real call = jy.zcInflationOption(Option::Call, k, T);
            Real put = jy.zcInflationOption(Option::Put, k, T);
            Real expected = std::exp(-0.03 * T) * (jy.forwardIndexRatio(T) - k);
            QL_CHECK_CLOSE(call - put, expected, 1.0e-10);
        }
    }
}

BOOST_AUTO_TEST_CASE(testZcOptionStandardDeviation) {
    BOOST_TEST_MESSAGE(
        "Testing Jarrow-Yildirim zero-coupon option against lognormal expectation...");
    Setup s;
    JarrowYildirim jy = s.model();
    const Real T = 5.0, strike = 1.1;
    // total variance from direct quadrature of the integrand, independent of the model
    Real variance = integrate(
        [&](Real u) -> Real {
            Real bn = bFun(s.an, T - u), br = bFun(s.ar, T - u);
            return s.sI * s.sI + s.sr * s.sr * br * br + s.sn * s.sn * bn * bn -
                   2.0 * s.rhoRI * s.sI * s.sr * br + 2.0 * s.rhoNI * s.sI * s.sn * bn -
                   2.0 * s.rhoNR * s.sn * s.sr * bn * br;
        },
        0.0, T);
    Real sd = std::sqrt(variance);
    Real forward = jy.forwardIndexRatio(T);
    Real df = std::exp(-0.03 * T);
    NormalDistribution density;
    // payoff is smooth on each side of the exercise boundary in the normal variate
    Real zStar = (std::log(strike / forward) + 0.5 * variance) / sd;
    auto payoff = [&](Real z) -> Real {
        return (forward * std::exp(sd * z - 0.5 * variance) - strike) * density(z);
    };
    Real callExpected = df * integrate(payoff, zStar, 12.0);
    Real putExpected = -df * integrate(payoff, -12.0, zStar);
    Real call = jy.zcInflationOption(Option::Call, strike, T);
    Real put = jy.zcInflationOption(Option::Put, strike, T);
    QL_CHECK_CLOSE(call, callExpected, 1.0e-6);
    QL_CHECK_CLOSE(put, putExpected, 1.0e-6);
    QL_CHECK_CLOSE(call, df * blackFormula(Option::Call, strike, forward, sd), 1.0e-9);
}

BOOST_AUTO_TEST_CASE(testYoYConvexityAgainstMonteCarlo) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim YoY forward against Monte Carlo...");
    Setup s;
    JarrowYildirim jy = s.model();
    const Real S = 4.0, T = 6.0;
    const Size samples = 1000000;

    // Exact joint law of the Ornstein-Uhlenbeck states at S under the
    // nominal T-forward measure: the real state carries the index-correlation
    // drift and both states carry the forward-measure shift.
    auto shiftIntegral = [&](Real a) -> Real {
        return integrate([&](Real u) -> Real { return std::exp(-a * (S - u)) * bFun(s.an, T - u); },
                         0.0, S);
    };
    Real meanN = -s.sn * s.sn * shiftIntegral(s.an);
    Real meanR =
        -s.rhoRI * s.sr * s.sI * bFun(s.ar, S) - s.sr * s.sn * s.rhoNR * shiftIntegral(s.ar);
    Real sdN = s.sn * std::sqrt((1.0 - std::exp(-2.0 * s.an * S)) / (2.0 * s.an));
    Real sdR = s.sr * std::sqrt((1.0 - std::exp(-2.0 * s.ar * S)) / (2.0 * s.ar));
    Real cov = s.rhoNR * s.sn * s.sr * (1.0 - std::exp(-(s.an + s.ar) * S)) / (s.an + s.ar);
    Real rho = cov / (sdN * sdR);

    HullWhite nominal(s.nominalCurve, s.an, s.sn), real(s.realCurve, s.ar, s.sr);
    auto nominalDyn = nominal.dynamics(), realDyn = real.dynamics();
    MersenneTwisterUniformRng rng(20261119);
    InverseCumulativeNormal inverse;
    Real sumY = 0.0, sumY2 = 0.0, sumInvP = 0.0, sumInvP2 = 0.0;
    for (Size i = 0; i < samples; ++i) {
        Real z1 = inverse(rng.next().value), z2 = inverse(rng.next().value);
        Real xn = meanN + sdN * z1;
        Real xr = meanR + sdR * (rho * z1 + std::sqrt(1.0 - rho * rho) * z2);
        Real pn = nominal.discountBond(S, T, nominalDyn->shortRate(S, xn));
        Real pr = real.discountBond(S, T, realDyn->shortRate(S, xr));
        Real y = pr / pn;
        sumY += y;
        sumY2 += y * y;
        sumInvP += 1.0 / pn;
        sumInvP2 += 1.0 / (pn * pn);
    }
    auto stats = [&](Real sum, Real sum2, Real& mean) -> Real {
        mean = sum / samples;
        return std::sqrt((sum2 / samples - mean * mean) / samples);
    };
    Real meanYoY, meanInvP;
    Real seYoY = stats(sumY, sumY2, meanYoY);
    Real seInvP = stats(sumInvP, sumInvP2, meanInvP);

    // E^T[1/P_n(S,T)] = P_n(0,S)/P_n(0,T) validates the forward-measure shift
    Real expectedInvP = std::exp(0.03 * (T - S));
    if (std::fabs(meanInvP - expectedInvP) > 4.0 * seInvP)
        BOOST_ERROR("E[1/Pn] " << meanInvP << " expected " << expectedInvP << " +- " << seInvP);

    Real calculated = jy.yoyForward(S, T);
    BOOST_TEST_MESSAGE("YoY forward: closed form " << calculated << ", Monte Carlo " << meanYoY
                                                   << " +- " << seYoY);
    Real curveOnly = jy.forwardIndexRatio(T) / jy.forwardIndexRatio(S);
    if (std::fabs(meanYoY - calculated) > 4.0 * seYoY)
        BOOST_ERROR("YoY forward " << calculated << " vs Monte Carlo " << meanYoY << " +- "
                                   << seYoY);
    // the simulation must be able to resolve the convexity adjustment
    if (std::fabs(calculated - curveOnly) < 20.0 * seYoY)
        BOOST_ERROR("convexity " << calculated - curveOnly << " not resolved by standard error "
                                 << seYoY);
}

BOOST_AUTO_TEST_CASE(testZcVarianceAgainstMonteCarlo) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim ZC variance against Monte Carlo...");
    Setup s;
    JarrowYildirim jy = s.model();
    const Real T = 5.0;
    const Size steps = 10, samples = 200000;
    const Real h = T / steps;

    // ln I_T = ln I_0 + int (r_n - r_r) dt - sigma_I^2 T/2 + sigma_I W_I(T): the
    // variance under the forward measure equals that under the risk-neutral
    // measure, since the measure change only shifts the mean. The rates are
    // deterministic plus Ornstein-Uhlenbeck states x with unit-time kernels
    // sigma e^{-a(h-u)} (state) and sigma B(h-u) (integral over the step), so
    // each step is sampled exactly from its joint Gaussian law
    // (xi_n, eta_n, xi_r, eta_r, omega).
    typedef std::function<Real(Real)> Kernel;
    std::vector<Kernel> kernels = {[&](Real u) -> Real { return s.sn * std::exp(-s.an * (h - u)); },
                                   [&](Real u) -> Real { return s.sn * bFun(s.an, h - u); },
                                   [&](Real u) -> Real { return s.sr * std::exp(-s.ar * (h - u)); },
                                   [&](Real u) -> Real { return s.sr * bFun(s.ar, h - u); },
                                   [](Real) -> Real { return 1.0; }};
    // driving Brownian motion of each kernel: 0 = nominal, 1 = real, 2 = index
    const Size driver[] = {0, 0, 1, 1, 2};
    const Real rho[3][3] = {
        {1.0, s.rhoNR, s.rhoNI}, {s.rhoNR, 1.0, s.rhoRI}, {s.rhoNI, s.rhoRI, 1.0}};
    Matrix cov(5, 5);
    for (Size i = 0; i < 5; ++i)
        for (Size j = 0; j < 5; ++j)
            cov[i][j] =
                rho[driver[i]][driver[j]] *
                integrate([&](Real u) -> Real { return kernels[i](u) * kernels[j](u); }, 0.0, h);
    Matrix chol = pseudoSqrt(cov, SalvagingAlgorithm::None);

    MersenneTwisterUniformRng rng(1119);
    InverseCumulativeNormal inverse;
    const Real decayN = std::exp(-s.an * h), decayR = std::exp(-s.ar * h);
    const Real stepN = bFun(s.an, h), stepR = bFun(s.ar, h);
    Real sum = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    for (Size i = 0; i < samples; ++i) {
        Real xn = 0.0, xr = 0.0, z = 0.0;
        for (Size k = 0; k < steps; ++k) {
            Real g[5];
            for (Real& gi : g)
                gi = inverse(rng.next().value);
            Real v[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
            for (Size a = 0; a < 5; ++a)
                for (Size b = 0; b <= a; ++b)
                    v[a] += chol[a][b] * g[b];
            z += stepN * xn + v[1] - stepR * xr - v[3] + s.sI * v[4];
            xn = decayN * xn + v[0];
            xr = decayR * xr + v[2];
        }
        sum += z;
        sum2 += z * z;
        sum3 += z * z * z;
        sum4 += z * z * z * z;
    }
    Real mean = sum / samples, var = sum2 / samples - mean * mean;
    // standard error of the sample variance from the fourth central moment
    Real m4 = sum4 / samples - 4.0 * mean * sum3 / samples + 6.0 * mean * mean * sum2 / samples -
              3.0 * mean * mean * mean * mean;
    Real se = std::sqrt((m4 - var * var) / samples);
    Real calculated = jy.zcInflationVariance(T);
    BOOST_TEST_MESSAGE("ZC variance: closed form " << calculated << ", Monte Carlo " << var
                                                   << " +- " << se);
    if (std::fabs(var - calculated) > 4.0 * se)
        BOOST_ERROR("ZC variance " << calculated << " vs Monte Carlo " << var << " +- " << se);
}

BOOST_AUTO_TEST_CASE(testParameterValidation) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim parameter validation...");
    Setup s;
    auto make = [&](Real an, Real sn, Real ar, Real sr, Real sI, Real rhoNR, Real rhoNI,
                    Real rhoRI) {
        JarrowYildirim(s.nominalCurve, s.realCurve, an, sn, ar, sr, sI, rhoNR, rhoNI, rhoRI);
    };
    auto correlations = [&](Real rhoNR, Real rhoNI, Real rhoRI) {
        make(s.an, s.sn, s.ar, s.sr, s.sI, rhoNR, rhoNI, rhoRI);
    };
    BOOST_CHECK_EXCEPTION(correlations(1.2, 0.0, 0.0), Error,
                          ExpectedErrorMessage("correlation 1.2 outside [-1, 1]"));
    BOOST_CHECK_EXCEPTION(correlations(0.0, -1.2, 0.0), Error,
                          ExpectedErrorMessage("correlation -1.2 outside [-1, 1]"));
    BOOST_CHECK_EXCEPTION(
        correlations(0.9, 0.9, 0.0), Error,
        ExpectedErrorMessage("correlation matrix is not positive semi-definite (determinant "));
    BOOST_CHECK_EXCEPTION(
        correlations(0.9, 0.9, -0.9), Error,
        ExpectedErrorMessage("correlation matrix is not positive semi-definite (determinant "));
    BOOST_CHECK_NO_THROW(correlations(1.0, 1.0, 1.0));
    BOOST_CHECK_EXCEPTION(make(0.0, s.sn, s.ar, s.sr, s.sI, 0.0, 0.0, 0.0), Error,
                          ExpectedErrorMessage("nominal mean reversion must be positive"));
    BOOST_CHECK_EXCEPTION(make(s.an, s.sn, 0.0, s.sr, s.sI, 0.0, 0.0, 0.0), Error,
                          ExpectedErrorMessage("real mean reversion must be positive"));
    BOOST_CHECK_EXCEPTION(make(s.an, -0.1, s.ar, s.sr, s.sI, 0.0, 0.0, 0.0), Error,
                          ExpectedErrorMessage("negative nominal volatility"));
    BOOST_CHECK_EXCEPTION(make(s.an, s.sn, s.ar, -0.1, s.sI, 0.0, 0.0, 0.0), Error,
                          ExpectedErrorMessage("negative real volatility"));
    BOOST_CHECK_EXCEPTION(make(s.an, s.sn, s.ar, s.sr, -0.1, 0.0, 0.0, 0.0), Error,
                          ExpectedErrorMessage("negative index volatility"));

    Handle<YieldTermStructure> empty;
    BOOST_CHECK_EXCEPTION(
        JarrowYildirim(empty, s.realCurve, s.an, s.sn, s.ar, s.sr, s.sI, 0.0, 0.0, 0.0), Error,
        ExpectedErrorMessage("empty nominal curve"));
    BOOST_CHECK_EXCEPTION(
        JarrowYildirim(s.nominalCurve, empty, s.an, s.sn, s.ar, s.sr, s.sI, 0.0, 0.0, 0.0), Error,
        ExpectedErrorMessage("empty real curve"));
    Handle<YieldTermStructure> shifted(
        ext::make_shared<FlatForward>(s.today + 1, Handle<Quote>(s.realRate), Actual365Fixed()));
    BOOST_CHECK_EXCEPTION(
        JarrowYildirim(s.nominalCurve, shifted, s.an, s.sn, s.ar, s.sr, s.sI, 0.0, 0.0, 0.0), Error,
        ExpectedErrorMessage("nominal and real curves must have the same reference date"));
    Handle<YieldTermStructure> otherDayCounter(
        ext::make_shared<FlatForward>(s.today, Handle<Quote>(s.realRate), Actual360()));
    BOOST_CHECK_EXCEPTION(
        JarrowYildirim(s.nominalCurve, otherDayCounter, s.an, s.sn, s.ar, s.sr, s.sI, 0.0, 0.0,
                       0.0),
        Error, ExpectedErrorMessage("nominal and real curves must have the same day counter"));

    // time arguments
    JarrowYildirim jy = s.model();
    BOOST_CHECK_EXCEPTION(jy.forwardIndexRatio(-1.0), Error,
                          ExpectedErrorMessage("negative time (-1) given"));
    BOOST_CHECK_EXCEPTION(jy.zcInflationVariance(-1.0), Error,
                          ExpectedErrorMessage("negative time (-1) given"));
    BOOST_CHECK_EXCEPTION(jy.zcInflationOption(Option::Call, 1.0, 0.0), Error,
                          ExpectedErrorMessage("non-positive time (0) given"));
    BOOST_CHECK_EXCEPTION(jy.zcInflationOption(Option::Call, 1.0, -1.0), Error,
                          ExpectedErrorMessage("non-positive time (-1) given"));
    BOOST_CHECK_EXCEPTION(jy.yoyConvexity(-1.0, 2.0), Error,
                          ExpectedErrorMessage("negative time (-1) given"));
    BOOST_CHECK_EXCEPTION(jy.yoyConvexity(3.0, 3.0), Error,
                          ExpectedErrorMessage("start time (3) must be smaller than end time (3)"));
    BOOST_CHECK_EXCEPTION(jy.yoyConvexity(4.0, 3.0), Error,
                          ExpectedErrorMessage("start time (4) must be smaller than end time (3)"));
    BOOST_CHECK_NO_THROW(jy.forwardIndexRatio(0.0));
    BOOST_CHECK_NO_THROW(jy.zcInflationVariance(0.0));
    BOOST_CHECK_NO_THROW(jy.yoyConvexity(0.0, 1.0));
}

BOOST_AUTO_TEST_CASE(testCurveConventionsAfterRelinking) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim curve conventions after relinking...");
    Setup s;
    RelinkableHandle<YieldTermStructure> real(*s.realCurve);
    JarrowYildirim jy(s.nominalCurve, real, s.an, s.sn, s.ar, s.sr, s.sI, s.rhoNR, s.rhoNI,
                      s.rhoRI);
    BOOST_CHECK_NO_THROW(jy.forwardIndexRatio(2.0));
    real.linkTo(
        ext::make_shared<FlatForward>(s.today + 1, Handle<Quote>(s.realRate), Actual365Fixed()));
    ExpectedErrorMessage message("nominal and real curves must have the same reference date");
    BOOST_CHECK_EXCEPTION(jy.forwardIndexRatio(2.0), Error, message);
    BOOST_CHECK_EXCEPTION(jy.zcInflationVariance(2.0), Error, message);
    BOOST_CHECK_EXCEPTION(jy.zcInflationOption(Option::Call, 1.0, 2.0), Error, message);
    BOOST_CHECK_EXCEPTION(jy.yoyConvexity(1.0, 2.0), Error, message);
    BOOST_CHECK_EXCEPTION(jy.yoyForward(1.0, 2.0), Error, message);
    real.linkTo(ext::make_shared<FlatForward>(s.today, Handle<Quote>(s.realRate), Actual360()));
    BOOST_CHECK_EXCEPTION(
        jy.forwardIndexRatio(2.0), Error,
        ExpectedErrorMessage("nominal and real curves must have the same day counter"));
}

BOOST_AUTO_TEST_CASE(testObserverRegistration) {
    BOOST_TEST_MESSAGE("Testing Jarrow-Yildirim observer notifications...");
    Setup s;
    auto jy = ext::make_shared<JarrowYildirim>(s.model());
    Flag flag;
    flag.registerWith(jy);
    Real before = jy->forwardIndexRatio(5.0);
    s.realRate->setValue(0.02);
    if (!flag.isUp())
        BOOST_ERROR("observer was not notified of a real curve change");
    Real afterReal = jy->forwardIndexRatio(5.0);
    if (afterReal >= before)
        BOOST_ERROR("forward index ratio did not fall after a real rate increase");
    flag.lower();
    s.nominalRate->setValue(0.035);
    if (!flag.isUp())
        BOOST_ERROR("observer was not notified of a nominal curve change");
    if (jy->forwardIndexRatio(5.0) <= afterReal)
        BOOST_ERROR("forward index ratio did not rise after a nominal rate increase");
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
