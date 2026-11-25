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

#include "toplevelfixture.hpp"
#include "utilities.hpp"
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/integrals/gausslobattointegral.hpp>
#include <ql/math/randomnumbers/rngtraits.hpp>
#include <ql/math/statistics/generalstatistics.hpp>
#include <ql/methods/finitedifferences/utilities/cevrndcalculator.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/vanilla/analyticcevengine.hpp>
#include <ql/pricingengines/vanilla/fdcevvanillaengine.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace QuantLib;
using boost::unit_test_framework::test_suite;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(FdCevTests)

class ExpectationFct {
  public:
    ExpectationFct(const CEVRNDCalculator& calculator, Time t)
    : t_(t), calculator_(calculator) { }

    Real operator()(Real f) const { return f*calculator_.pdf(f, t_); }

  private:
    const Time t_;
    const CEVRNDCalculator& calculator_;
};


BOOST_AUTO_TEST_CASE(testLocalMartingale) {
    BOOST_TEST_MESSAGE(
        "Testing local martingale property of CEV process with PDF...");

    const Time t = 1.0;

    const Real f0 = 2.1;
    const Real alpha = 1.75;
    const Real betas[] = {-2.4, 0.23, 0.9, 1.1, 1.5};

    for (Real beta : betas) {
        const CEVRNDCalculator rndCalculator(f0, alpha, beta);

        const Real eps = 1e-10;
        const Real tol = 100*eps;

        const Real upperBound = 10*rndCalculator.invcdf(1-eps, t);

        const Real expectationValue = GaussLobattoIntegral(10000, eps)(
            ExpectationFct(rndCalculator, t), QL_EPSILON, upperBound);

        const Real diff = expectationValue-f0;


        if (beta < 1.0 && std::fabs(diff) > tol) {
            BOOST_ERROR("CEV process should be a martingale for beta < 1.0"
                        << "\n    expected:   " << f0
                        << std::scientific
                        << "\n    difference  " << diff
                        << "\n    tolerance:  " << tol);
        }

        if (beta > 1.0 && diff > -tol) {
            BOOST_ERROR("CEV process should only be a local martingale "
                        "for beta > 1.0. Expectation is E[F_t|F_0] < F_0"
                        << "\n    E[F_t|F_0]: " << expectationValue
                        << "\n    F_0:        " << f0);
        }

        // check local martingale property with Monte-Carlo simulation
        const Size nSims = 5000;

        const Size nSteps = 2000;
        const Real dt = t / nSteps;
        const Real sqrtDt = std::sqrt(dt);

        GeneralStatistics stat;
        const PseudoRandom::rng_type mt(MersenneTwisterUniformRng(42));

        if (beta > 1.2) {
            for (Size i=0; i < nSims; ++i) {
                Real f = f0;
                for (Size j=0; j < nSteps; ++j) {
                    f += alpha * std::pow(f, beta) * mt.next().value * sqrtDt;
                    f = std::max(0.0, f);

                    if (f == 0.0) break; // absorbing boundary
                }
                stat.add(f - f0);
            }

            const Real calculated = stat.mean();
            const Real error = stat.errorEstimate();

            if (std::fabs(calculated - diff) > 2.35*error) {
                BOOST_ERROR(
                    "failed to calculate local martingale property "
                    "by Monte-Carlo Simulation for beta > 1.0. "
                            << "\n    E[F_t|F_0]   : " << expectationValue
                            << "\n    E_MC[F_t|F_0]: " << calculated + f0
                            << "\n    error_MC     : " << error
                            << "\n    difference   : " << std::fabs(calculated - diff)
                            << "\n    tolerance    : " << 2.35*error);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testFdmCevOp) {
    BOOST_TEST_MESSAGE(
            "Testing FDM constant elasticity of variance (CEV) operator...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Date maturityDate = today + Period(12, Months);
    const Real strike = 2.3;

    const Option::Type optionTypes[] = { Option::Call, Option::Put};

    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturityDate);

    for (auto optionType : optionTypes) {
        const ext::shared_ptr<PlainVanillaPayoff> payoff =
            ext::make_shared<PlainVanillaPayoff>(optionType, strike);

        const ext::shared_ptr<YieldTermStructure> rTS =
            flatRate(today, 0.15, dc);

        const Real f0 = 2.1;
        const Real alpha = 0.75;

        const Real betas[] = { -2.0, -0.5, 0.45, 0.6, 0.9, 1.45 };
        for (Real beta : betas) {

            VanillaOption option(payoff, exercise);
            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0, alpha, beta, Handle<YieldTermStructure>(rTS)));

            const Real analyticNPV = option.NPV();

            const Real eps = 1e-3;

            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0*(1+eps), alpha, beta, Handle<YieldTermStructure>(rTS)));
            const Real analyticUpNPV = option.NPV();

            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0*(1-eps), alpha, beta, Handle<YieldTermStructure>(rTS)));
            const Real analyticDownNPV = option.NPV();

            const Real analyticDelta = (analyticUpNPV - analyticDownNPV)
                /(2*eps*f0);

            option.setPricingEngine(ext::make_shared<FdCEVVanillaEngine>(
                f0, alpha, beta, Handle<YieldTermStructure>(rTS),
                100, 1000, 1, 1.0, 1e-6));

            const Real calculatedNPV = option.NPV();
            const Real calculatedDelta = option.delta();

            const Real tol = 0.01;
            if (std::fabs(calculatedNPV - analyticNPV) > tol
                || std::fabs(calculatedDelta - analyticDelta) > tol) {
                BOOST_ERROR(
                    "failed to calculate vanilla option prices/delta "
                    << "\n    beta            : " << beta
                    << "\n    option type     : "
                    << ((payoff->optionType() == Option::Call) ? "Call" : "Put")
                    << "\n    analytic npv    : " << analyticNPV
                    << "\n    pde npv         : " << calculatedNPV
                    << "\n    npv difference  : "
                    << std::fabs(calculatedNPV - analyticNPV)
                    << "\n    tolerance       : " << tol
                    << "\n    analytic delta  : " << analyticDelta
                    << "\n    pde delta       : " << calculatedDelta
                    << "\n    delta difference: "
                    << std::fabs(calculatedDelta - analyticDelta)
                    << "\n    tolerance       : " << tol);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testAnalyticCevPutCallParity) {
    BOOST_TEST_MESSAGE(
        "Testing put-call parity of the analytic CEV engine...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Date maturityDate = today + Period(12, Months);
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturityDate);

    const Real f0 = 100.0, alpha = 0.30;
    const Real betas[] = {-0.5, 0.0, 0.45, 0.9, 0.99, 1.1, 1.5};
    const Real strikes[] = {80.0, 100.0, 120.0};
    const Real rates[] = {0.03, 0.15};

    for (Real rate : rates) {
        const ext::shared_ptr<YieldTermStructure> rTS =
            flatRate(today, rate, dc);

        for (Real beta : betas) {
            for (Real strike : strikes) {
                VanillaOption call(
                    ext::make_shared<PlainVanillaPayoff>(Option::Call, strike),
                    exercise);
                VanillaOption put(
                    ext::make_shared<PlainVanillaPayoff>(Option::Put, strike),
                    exercise);

                call.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                    f0, alpha, beta, Handle<YieldTermStructure>(rTS)));
                put.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                    f0, alpha, beta, Handle<YieldTermStructure>(rTS)));

                // call - put = disc*(E[f_T] - strike), where the
                // discounted expectation E[f_T*disc] equals f0 for
                // beta <= 1 (true martingale starting from f0) and
                // f0*gamma_p(delta/2-1, x0/(2*tau)) for beta > 1 (strict
                // local martingale, bounded by f0 but not by f0*disc),
                // evaluated at the effective time tau of the drifted
                // process.
                const DiscountFactor discount = rTS->discount(maturityDate);
                Real expectedForward = f0 - strike*discount;
                if (beta > 1.0) {
                    const Time tau = cevEffectiveTime(
                        alpha, beta, rTS, 0.0,
                        rTS->timeFromReference(maturityDate));
                    const Real delta = (1.0-2.0*beta)/(1.0-beta);
                    const Real x0 = std::pow(f0, 2.0*(1.0-beta))
                        /((1.0-beta)*(1.0-beta));

                    expectedForward = f0*boost::math::gamma_p(
                        0.5*delta-1.0, x0/(2.0*tau)) - strike*discount;
                }

                const Real residual =
                    call.NPV() - put.NPV() - expectedForward;
                const Real tol = 1e-8*(f0+strike);

                if (std::fabs(residual) > tol) {
                    BOOST_ERROR(
                        "put-call parity violated"
                        << "\n    rate            : " << rate
                        << "\n    beta            : " << beta
                        << "\n    strike          : " << strike
                        << "\n    call npv        : " << call.NPV()
                        << "\n    put npv         : " << put.NPV()
                        << "\n    residual        : " << residual
                        << "\n    tolerance       : " << tol);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testAnalyticCevBlackLimit) {
    BOOST_TEST_MESSAGE(
        "Testing the Black limit of the analytic CEV engine at beta=1...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Date maturityDate = today + Period(12, Months);
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturityDate);

    const Real f0 = 100.0, alpha = 0.30;
    const ext::shared_ptr<YieldTermStructure> rTS =
        flatRate(today, 0.03, dc);
    const DiscountFactor discount = rTS->discount(maturityDate);
    const Real stdDev = alpha * std::sqrt(
        rTS->timeFromReference(maturityDate));

    const Option::Type optionTypes[] = { Option::Call, Option::Put };
    const Real strikes[] = {80.0, 100.0, 120.0};

    for (auto optionType : optionTypes) {
        for (Real strike : strikes) {
            const ext::shared_ptr<PlainVanillaPayoff> payoff =
                ext::make_shared<PlainVanillaPayoff>(optionType, strike);
            const Real black = blackFormula(
                payoff, f0/discount, stdDev, discount);

            for (Real beta : {1.0-1e-6, 1.0-1e-4, 1.0, 1.0+1e-4, 1.0+1e-6}) {
                VanillaOption option(payoff, exercise);
                option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                    f0, alpha, beta, Handle<YieldTermStructure>(rTS)));

                const Real calculated = option.NPV();

                // exact at beta=1; within model curvature of the limit
                // for the neighbouring beta values
                const Real tol = (beta == 1.0) ? Real(1e-10*black) : Real(0.05);

                if (std::fabs(calculated - black) > tol) {
                    BOOST_ERROR(
                        "failed to reproduce the Black limit"
                        << "\n    beta            : " << beta
                        << "\n    option type     : "
                        << ((optionType == Option::Call) ? "Call" : "Put")
                        << "\n    strike          : " << strike
                        << "\n    black price     : " << black
                        << "\n    calculated      : " << calculated
                        << "\n    difference      : "
                        << std::fabs(calculated - black)
                        << "\n    tolerance       : " << tol);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testAnalyticCevKnownValues) {
    BOOST_TEST_MESSAGE(
        "Testing analytic CEV engine values against closed forms and "
        "Monte Carlo anchors...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Date maturityDate = today + Period(12, Months);
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturityDate);

    const Real f0 = 100.0, alpha = 0.30;
    const Real rate = 0.03;
    const ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rate, dc);
    const DiscountFactor discount = rTS->discount(maturityDate);
    const Time maturity = rTS->timeFromReference(maturityDate);

    // 1. Scholes (1976) closed form for the absolute diffusion
    //    dS = rS dt + alpha dW, i.e. beta = 0
    {
        // a strike near the forward keeps d moderate, so the volatility
        // dependent term of the closed form stays material at the tolerance
        const Real strike = 103.0;
        const Real forward = f0/discount;
        const Real sqrtV = alpha*std::sqrt(
            (std::exp(2*rate*maturity)-1.0)/(2*rate));
        const Real d = (forward - strike)/sqrtV;
        const Real expected = discount*((forward - strike)
            * CumulativeNormalDistribution()(d)
            + sqrtV*NormalDistribution()(d));

        VanillaOption option(
            ext::make_shared<PlainVanillaPayoff>(Option::Call, strike),
            exercise);
        option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
            f0, alpha, 0.0, Handle<YieldTermStructure>(rTS)));

        const Real calculated = option.NPV();
        if (std::fabs(calculated - expected) > 1e-5) {
            BOOST_ERROR(
                "failed to reproduce the Scholes closed form at beta=0"
                << "\n    expected        : " << expected
                << "\n    calculated      : " << calculated
                << "\n    difference      : "
                << std::fabs(calculated - expected));
        }
    }

    // 2. Monte Carlo anchors. Euler discretization with antithetic paths
    //    and an exact GBM control variate, 2x200000 paths, seed 20261125;
    //    the standard error of each value is below 0.0005.
    {
        const Real mcValues[][2] = {
            { 0.999, 13.2300 },
            { 0.99,  12.7611 },
            { 0.95,  10.8955 },
            { 0.9,    8.9994 },
            { 0.5,    3.2017 },
            { 1.1,   20.0291 }
        };

        for (const auto& mcValue : mcValues) {
            const Real beta = mcValue[0];

            VanillaOption option(
                ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                exercise);
            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0, alpha, beta, Handle<YieldTermStructure>(rTS)));

            const Real calculated = option.NPV();
            if (std::fabs(calculated - mcValue[1]) > 0.005) {
                BOOST_ERROR(
                    "failed to reproduce the Monte Carlo anchor value"
                    << "\n    beta            : " << beta
                    << "\n    expected        : " << mcValue[1]
                    << "\n    calculated      : " << calculated
                    << "\n    difference      : "
                    << std::fabs(calculated - mcValue[1])
                    << "\n    tolerance       : 0.005");
            }
        }
    }

    // 3. zero rate invariance: at zero rates the drifted transform
    //    reduces to the plain Schroder noncentral chi-squared formula
    {
        const ext::shared_ptr<YieldTermStructure> rTS0 =
            flatRate(today, 0.0, dc);
        const Real r0Values[][3] = {
            { 0.999,  90.0, 16.9649458981 },
            { 0.999, 110.0,  8.08624096975 },
            { 0.95,   90.0, 14.8930447001 },
            { 0.95,  100.0,  9.48434259877 },
            { 0.95,  110.0,  5.71842357822 },
            { 0.9,   100.0,  7.54031962747 },
            { 0.8,   100.0,  4.76193587653 },
            { 0.5,   100.0,  1.19679317903 },
            { 0.2,   100.0,  0.300629054252 },
            { 0.0,   100.0,  0.11968268412 },
            { -0.5,  100.0,  0.0119682684681 },
            { 1.1,   100.0, 18.7928971285 },
            { 1.5,   100.0,  1.85310127805 }
        };

        for (const auto& r0Value : r0Values) {
            const Real beta = r0Value[0];
            const Real strike = r0Value[1];

            VanillaOption option(
                ext::make_shared<PlainVanillaPayoff>(Option::Call, strike),
                exercise);
            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0, alpha, beta, Handle<YieldTermStructure>(rTS0)));

            const Real calculated = option.NPV();
            if (std::fabs(calculated - r0Value[2]) > 1e-9) {
                BOOST_ERROR(
                    "zero-rate value differs from the driftless "
                    "Schroder value"
                    << "\n    beta            : " << beta
                    << "\n    strike          : " << strike
                    << "\n    expected        : " << r0Value[2]
                    << "\n    calculated      : " << calculated
                    << "\n    difference      : "
                    << std::fabs(calculated - r0Value[2]));
            }
        }

        // a direct beta > 1 put value, independently verified by pdf
        // quadrature: for beta > 1 the process is a strict local martingale
        // with E[f_T] = f0*gamma_p(delta/2-1, x0/(2t)) < f0, so the put
        // is large even at the money
        {
            VanillaOption option(
                ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
                exercise);
            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0, alpha, 1.5, Handle<YieldTermStructure>(rTS0)));

            const Real calculated = option.NPV();
            if (std::fabs(calculated - 81.9268415697) > 1e-6) {
                BOOST_ERROR(
                    "beta > 1 put value differs from the independently "
                    "verified value"
                    << "\n    expected        : " << 81.9268415697
                    << "\n    calculated      : " << calculated
                    << "\n    difference      : "
                    << std::fabs(calculated - 81.9268415697)
                    << "\n    tolerance       : 1e-6");
            }
        }
    }

    // 4. the engine's quadrature of the effective time under a non-zero
    //    flat rate is checked against the closed form of the integral
    {
        const Real strike = 100.0;

        for (Real beta : {0.5, 1.1}) {
            const Time effectiveTime = alpha*alpha
                *(1.0 - std::exp(-2.0*(1.0-beta)*rate*maturity))
                /(2.0*(1.0-beta)*rate);

            const CEVCalculator unitVolCalculator(f0, 1.0, beta);
            const Real expected = unitVolCalculator.value(
                Option::Call, strike*discount, effectiveTime);

            VanillaOption option(
                ext::make_shared<PlainVanillaPayoff>(Option::Call, strike),
                exercise);
            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0, alpha, beta, Handle<YieldTermStructure>(rTS)));

            const Real calculated = option.NPV();
            if (std::fabs(calculated - expected) > 1e-9) {
                BOOST_ERROR(
                    "failed to integrate the effective time over a flat rate"
                    << "\n    beta            : " << beta
                    << "\n    expected        : " << expected
                    << "\n    calculated      : " << calculated
                    << "\n    difference      : "
                    << std::fabs(calculated - expected)
                    << "\n    tolerance       : 1e-9");
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testFdmCevPiecewiseRateCurve) {
    BOOST_TEST_MESSAGE(
        "Testing the FDM CEV engine on a rate curve ending at maturity...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Date maturityDate = today + Period(12, Months);
    const Date stepDate = today + Period(6, Months);

    // the curve's maximum time is the option maturity: the boundary
    // condition's discount queries must stay within the curve's range
    const std::vector<Date> curveDates = { today, stepDate, maturityDate };
    const std::vector<DiscountFactor> curveDiscounts = {
        1.0, std::exp(-0.02*0.5), std::exp(-0.02*0.5 - 0.08*0.5)
    };
    const Handle<YieldTermStructure> rTS(
        ext::make_shared<DiscountCurve>(curveDates, curveDiscounts, dc));

    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturityDate);

    const Real f0 = 100.0, alpha = 0.30, strike = 100.0;

    VanillaOption option(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, strike), exercise);
    option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
        f0, alpha, 0.9, rTS));
    const Real analyticNPV = option.NPV();

    // a time grid finer than one day (1/365 of a year) per step drives
    // the rollback's boundary evaluation up against the curve's
    // maximum time
    option.setPricingEngine(ext::make_shared<FdCEVVanillaEngine>(
        f0, alpha, 0.9, rTS, 400, 400, 1, 1.0, 1e-6));

    const Real calculatedNPV = option.NPV();
    const Real tol = 0.01;
    if (std::fabs(calculatedNPV - analyticNPV) > tol) {
        BOOST_ERROR(
            "failed to calculate the vanilla option price on a curve "
            "ending at maturity"
            << "\n    analytic npv    : " << analyticNPV
            << "\n    pde npv         : " << calculatedNPV
            << "\n    npv difference  : "
            << std::fabs(calculatedNPV - analyticNPV)
            << "\n    tolerance       : " << tol);
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
