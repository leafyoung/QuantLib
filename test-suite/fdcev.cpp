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
#include <ql/methods/finitedifferences/meshers/fdmcev1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/meshers/uniform1dmesher.hpp>
#include <ql/methods/finitedifferences/operators/fdmcevop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/utilities/cevrndcalculator.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/vanilla/analyticcevengine.hpp>
#include <ql/pricingengines/vanilla/fdcevvanillaengine.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/termstructures/yield/discountcurve.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/time/daycounters/actualactual.hpp>
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
        const auto payoff =
            ext::make_shared<PlainVanillaPayoff>(optionType, strike);

        const auto rTS =
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

                call.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
                    f0, alpha, beta, Handle<YieldTermStructure>(rTS)));
                put.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
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
                        Real(0.5*delta-1.0), Real(x0/(2.0*tau))) - strike*discount;
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
    const Time maturity = rTS->timeFromReference(maturityDate);

    const Option::Type optionTypes[] = { Option::Call, Option::Put };
    const Real strikes[] = {80.0, 100.0, 120.0};

    for (auto optionType : optionTypes) {
        for (Real strike : strikes) {
            const ext::shared_ptr<PlainVanillaPayoff> payoff =
                ext::make_shared<PlainVanillaPayoff>(optionType, strike);
            for (Real beta : std::vector<Real>{
                     1.0-5e-5, 1.0-1e-6, 1.0, 1.0+1e-6, 1.0+5e-5}) {
                // Black price at the local lognormal volatility f0^(beta-1)
                // of the unit-volatility process run for the effective time
                const Real black = blackFormula(
                    payoff, f0/discount,
                    std::pow(f0, beta-1.0)
                        *std::sqrt(cevEffectiveTime(alpha, beta, rTS, 0.0, maturity)),
                    discount);
                VanillaOption option(payoff, exercise);
                option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
                    f0, alpha, beta, Handle<YieldTermStructure>(rTS)));

                const Real calculated = option.NPV();

                // inside the band the engine returns the Black price
                const Real tol = 1e-10*black;

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

                // the FDM boundary applies the same limit inside the band;
                // the PDE solution must reproduce it there as well. beta is
                // kept off one because the mesher's transition calculator
                // requires beta != 1
                if (beta == 1.0-5e-5) {
                    option.setPricingEngine(
                        ext::make_shared<FdDriftedCEVVanillaEngine>(
                            f0, alpha, beta, Handle<YieldTermStructure>(rTS),
                            100, 200, 1, 1.0, 1e-6));

                    const Real fdCalculated = option.NPV();
                    const Real fdTol = 0.01;

                    if (std::fabs(fdCalculated - black) > fdTol) {
                        BOOST_ERROR(
                            "failed to reproduce the Black limit with the "
                            "FDM engine"
                            << "\n    option type     : "
                            << ((optionType == Option::Call) ? "Call" : "Put")
                            << "\n    strike          : " << strike
                            << "\n    black price     : " << black
                            << "\n    calculated      : " << fdCalculated
                            << "\n    difference      : "
                            << std::fabs(fdCalculated - black)
                            << "\n    tolerance       : " << fdTol);
                    }
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
        option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
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
            option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
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
        // beta < 1 values: Schroder's formula evaluated with scipy.stats.ncx2
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
            option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
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
            option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
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

        for (Real beta : std::vector<Real>{0.5, 1.1}) {
            const Time effectiveTime = alpha*alpha
                *(1.0 - std::exp(-2.0*(1.0-beta)*rate*maturity))
                /(2.0*(1.0-beta)*rate);

            const CEVCalculator unitVolCalculator(f0, 1.0, beta);
            const Real expected = unitVolCalculator.value(
                Option::Call, strike*discount, effectiveTime);

            VanillaOption option(
                ext::make_shared<PlainVanillaPayoff>(Option::Call, strike),
                exercise);
            option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
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

        // at beta = 1 the effective time is the bare squared volatility
        // times the interval
        if (std::fabs(cevEffectiveTime(alpha, 1.0, rTS, 0.0, maturity)
                      - alpha*alpha*maturity) > 1e-12) {
            BOOST_ERROR(
                "the beta = 1 effective time is not alpha^2*(end-start)");
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
    option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
        f0, alpha, 0.9, rTS));
    const Real analyticNPV = option.NPV();

    // a time grid with more than 365 steps per year puts rollback steps
    // within a day of the curve's maximum time
    option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
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

BOOST_AUTO_TEST_CASE(testAnalyticCevExpiredOption) {
    BOOST_TEST_MESSAGE("Testing the analytic CEV engine on an expired "
        "option...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;
    Settings::instance().includeReferenceDateEvents() = true;

    const Handle<YieldTermStructure> rTS(flatRate(today, 0.03, dc));

    // the exercise date at the curve's reference date gives a zero
    // maturity: the engine rejects it
    VanillaOption option(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
        ext::make_shared<EuropeanExercise>(today));
    option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
        100.0, 0.30, 0.9, rTS));

    BOOST_CHECK_THROW(option.NPV(), QuantLib::Error);

    // the FDM engine rejects the same input through its own maturity path
    option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
        100.0, 0.30, 0.9, rTS, 100, 100, 1, 1.0, 1e-6));

    BOOST_CHECK_THROW(option.NPV(), QuantLib::Error);

    // inside the Black band the same input is rejected, not priced
    option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
        100.0, 0.30, 1.0, rTS));

    BOOST_CHECK_THROW(option.NPV(), QuantLib::Error);
}

BOOST_AUTO_TEST_CASE(testCevEffectiveTimeWithNonzeroStart) {
    BOOST_TEST_MESSAGE("Testing the CEV effective time on sub-intervals...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real alpha = 0.30, beta = 0.7, rate = 0.04;
    const Time start = 0.4, end = 1.3;

    // flat curve: closed form depending on the interval length only
    const ext::shared_ptr<YieldTermStructure> flat = flatRate(today, rate, dc);
    const Real k = 2.0*(1.0-beta);
    const Real flatExpected = alpha*alpha *
        (1.0 - std::exp(-k*rate*(end-start))) / (k*rate);
    const Real flatCalculated = cevEffectiveTime(alpha, beta, flat, start, end);
    if (std::fabs(flatCalculated - flatExpected) > 1e-12*flatExpected) {
        BOOST_ERROR("failed to reproduce the flat curve effective time"
                    << "\n    expected   : " << flatExpected
                    << "\n    calculated : " << flatCalculated);
    }

    // beta = 1: geometric Brownian motion, alpha^2 times the interval length
    const Real gbm = cevEffectiveTime(alpha, 1.0, flat, start, end);
    QL_CHECK_CLOSE(gbm, alpha*alpha*(end-start), 1e-10);

    BOOST_CHECK_THROW(cevEffectiveTime(alpha, beta, flat, end, end), Error);
    BOOST_CHECK_THROW(cevEffectiveTime(alpha, beta, flat, end, start), Error);

    // curve with a forward rate jump: compare with a fine midpoint sum.
    // The jump is a kink of the integrand, which limits the 64 point rule
    // on one panel to about 1e-6 relative accuracy here.
    const ext::shared_ptr<YieldTermStructure> kinked =
        ext::make_shared<DiscountCurve>(
            std::vector<Date>{today, today + Period(6, Months),
                              today + Period(24, Months)},
            std::vector<DiscountFactor>{
                1.0, std::exp(-0.01*0.5),
                std::exp(-0.01*0.5 - 0.12*1.5)}, dc);
    const Size n = 200000;
    const Real h = (end-start)/n;
    Real sum = 0.0;
    const DiscountFactor dStart = kinked->discount(start);
    for (Size i = 0; i < n; ++i)
        sum += std::pow(kinked->discount(start + (i+0.5)*h)/dStart, k);
    const Real kinkedExpected = alpha*alpha*h*sum;
    const Real kinkedCalculated =
        cevEffectiveTime(alpha, beta, kinked, start, end);
    if (std::fabs(kinkedCalculated - kinkedExpected) > 1e-5*kinkedExpected) {
        BOOST_ERROR("failed to reproduce the kinked curve effective time"
                    << "\n    expected   : " << kinkedExpected
                    << "\n    calculated : " << kinkedCalculated);
    }

    // long maturity with an early forward rate jump: the integral of
    // exp(-k*int r) has a closed form for a piecewise flat forward curve.
    // The kink at 3 years is resolved only if the rule is applied per year.
    const Real r1 = 0.01, r2 = 0.15, kink = 3.0, horizon = 30.0;
    const ext::shared_ptr<YieldTermStructure> longKinked =
        ext::make_shared<DiscountCurve>(
            std::vector<Date>{today, today + 365*3, today + 365*30},
            std::vector<DiscountFactor>{
                1.0, std::exp(-r1*kink),
                std::exp(-r1*kink - r2*(horizon-kink))}, dc);
    const Real longExpected =  // alpha = 1, beta = 0: k = 2
        (1.0 - std::exp(-2.0*r1*kink))/(2.0*r1)
        + std::exp(-2.0*r1*kink)
            *(1.0 - std::exp(-2.0*r2*(horizon-kink)))/(2.0*r2);
    QL_CHECK_CLOSE(cevEffectiveTime(1.0, 0.0, longKinked, 0.0, horizon),
                   longExpected, 1e-3);
}

BOOST_AUTO_TEST_CASE(testCevEnginesReactToRateCurve) {
    BOOST_TEST_MESSAGE("Testing that the CEV engines follow the rate curve...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const auto quote = ext::make_shared<SimpleQuote>(0.01);
    const Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today, Handle<Quote>(quote), dc));

    VanillaOption option(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
        ext::make_shared<EuropeanExercise>(today + Period(12, Months)));

    const std::vector<ext::shared_ptr<PricingEngine>> engines = {
        ext::make_shared<AnalyticDriftedCEVEngine>(100.0, 0.30, 0.8, rTS),
        ext::make_shared<FdDriftedCEVVanillaEngine>(100.0, 0.30, 0.8, rTS,
                                             50, 100, 1, 1.0, 1e-6)
    };

    for (const auto& engine : engines) {
        option.setPricingEngine(engine);
        const Real before = option.NPV();
        quote->setValue(0.06);
        const Real after = option.NPV();
        quote->setValue(0.01);

        // a higher rate raises the call price of the drifted asset by
        // several units here; a stale cached value would not move
        if (after - before < 1.0) {
            BOOST_ERROR("the price did not follow the rate curve"
                        << "\n    at 1% : " << before
                        << "\n    at 6% : " << after);
        }
    }
}

BOOST_AUTO_TEST_CASE(testFdmCevBlackBandBoundary) {
    BOOST_TEST_MESSAGE("Testing the FDM CEV boundary inside the Black band...");

    // A wide mesh (large eps) puts the upper Dirichlet boundary close
    // enough to the spot for its value to matter. Just inside the Black
    // band the boundary uses the Black limit; just outside, the closed
    // form. The two prices describe the same model to O(band width), so
    // they must agree whatever the boundary value is.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, alpha = 0.30, rate = 0.08;
    const Handle<YieldTermStructure> rTS(flatRate(today, rate, dc));
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + Period(12, Months));

    // beta = 1-1.5e-4 lies just outside the band. The tolerance covers the
    // genuine model difference between the two betas (about 0.005 here); a
    // wrong boundary value moves the price by much more.
    const Real eps = 0.2;
    const Real tol = 0.01;
    for (Option::Type type : {Option::Call, Option::Put}) {
        const auto payoff = ext::make_shared<PlainVanillaPayoff>(type, 100.0);
        VanillaOption vanilla(payoff, exercise);
        std::vector<Real> prices;
        for (Real beta : std::vector<Real>{1.0-5e-5, 1.0-1.5e-4}) {
            vanilla.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
                f0, alpha, beta, rTS, 200, 400, 1, 1.0, eps));
            prices.push_back(vanilla.NPV());
        }
        if (std::fabs(prices[0] - prices[1]) > tol) {
            BOOST_ERROR("the FDM boundary inside the Black band disagrees "
                        "with the closed form outside it"
                        << "\n    inside  : " << prices[0]
                        << "\n    outside : " << prices[1]
                        << "\n    tolerance: " << tol);
        }
    }
}

BOOST_AUTO_TEST_CASE(testFdmCevNearBlackLimitAwayFromTheMoney) {
    BOOST_TEST_MESSAGE("Testing the FDM CEV engine near beta = 1 away from the money...");

    // Close to beta = 1 the noncentral chi-squared evaluation at a
    // far-from-the-money boundary can fail. The engine must still price,
    // and agree with the analytic engine.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, alpha = 0.10;
    const Handle<YieldTermStructure> rTS(flatRate(today, 0.0, dc));
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + Period(12, Months));

    for (Real beta : std::vector<Real>{0.999, 1.001}) {
        for (Real strike : std::vector<Real>{70.0, 130.0}) {
            for (Option::Type type : {Option::Call, Option::Put}) {
                VanillaOption fd(
                    ext::make_shared<PlainVanillaPayoff>(type, strike), exercise);
                fd.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
                    f0, alpha, beta, rTS, 200, 400));
                VanillaOption analytic(
                    ext::make_shared<PlainVanillaPayoff>(type, strike), exercise);
                analytic.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
                    f0, alpha, beta, rTS));
                const Real expected = analytic.NPV();
                QL_CHECK_CLOSE(fd.NPV(), expected, 2.0);
                QL_CHECK_SMALL(fd.NPV() - expected, 5e-3);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testFdmCevBoundaryFallbackValue) {
    BOOST_TEST_MESSAGE("Testing the FDM CEV boundary value near beta = 1 on a wide mesh...");

    // A wide mesh (eps = 0.2) makes the boundary values material. For these
    // inputs the noncentral chi-squared evaluation fails at both boundaries,
    // so the price depends on the Black-Scholes boundary value.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, vol = 0.10, strike = 100.0;
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + 30);

    for (Real beta : std::vector<Real>{0.9998, 1.0002}) {
        for (Rate rate : std::vector<Rate>{-0.02, 0.05}) {
            for (Option::Type type : {Option::Call, Option::Put}) {
                const Handle<YieldTermStructure> rTS(flatRate(today, rate, dc));
                const Real alpha = vol*std::pow(f0, 1.0-beta);
                VanillaOption option(
                    ext::make_shared<PlainVanillaPayoff>(type, strike), exercise);
                option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
                    f0, alpha, beta, rTS));
                const Real expected = option.NPV();
                option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
                    f0, alpha, beta, rTS, 50, 400, 2, 1.0, 0.2));
                QL_CHECK_SMALL(option.NPV() - expected, 2e-3);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testCevEnginesRejectNonVanillaPayoffs) {
    BOOST_TEST_MESSAGE("Testing that the CEV engines reject non-vanilla payoffs...");

    // the boundary and closed-form values are vanilla prices, so a
    // digital payoff would be priced silently wrong
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Handle<YieldTermStructure> rTS(flatRate(today, 0.03, dc));
    VanillaOption digital(
        ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0),
        ext::make_shared<EuropeanExercise>(today + Period(12, Months)));

    digital.setPricingEngine(
        ext::make_shared<AnalyticDriftedCEVEngine>(100.0, 0.3, 0.5, rTS));
    BOOST_CHECK_THROW(digital.NPV(), Error);

    digital.setPricingEngine(
        ext::make_shared<FdDriftedCEVVanillaEngine>(100.0, 0.3, 0.5, rTS, 50, 100, 1));
    BOOST_CHECK_THROW(digital.NPV(), Error);
}

BOOST_AUTO_TEST_CASE(testFdmDriftedCevAbsorbingBoundary) {
    BOOST_TEST_MESSAGE(
        "Testing the drifted FDM CEV engine with an absorbing zero...");

    // For beta < 1 the process is absorbed at zero, and with a high
    // volatility the mesh reaches it: the lower boundary value is the
    // discounted payoff at zero, which dominates the put price here.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 10.0, vol = 1.0;
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + Period(24, Months));

    struct Case { Real beta, strike, rate; };
    for (const Case& c : {Case{0.0, 20.0, 0.0}, Case{0.3, 10.0, 0.0},
                          Case{0.6, 10.0, 0.05}}) {
        const Real alpha = vol*std::pow(f0, 1.0-c.beta);
        const Handle<YieldTermStructure> rTS(flatRate(today, c.rate, dc));
        VanillaOption option(
            ext::make_shared<PlainVanillaPayoff>(Option::Put, c.strike),
            exercise);
        option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
            f0, alpha, c.beta, rTS));
        const Real analytic = option.NPV();
        option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
            f0, alpha, c.beta, rTS, 100, 400, 2));
        QL_CHECK_SMALL(option.NPV() - analytic, 5e-3);
    }
}

BOOST_AUTO_TEST_CASE(testDriftedAnalyticEngineRejectsAmericanExercise) {
    BOOST_TEST_MESSAGE(
        "Testing that the drifted analytic CEV engine rejects early exercise...");

    const Date today = Date(22, February, 2018);
    Settings::instance().evaluationDate() = today;

    const Handle<YieldTermStructure> rTS(flatRate(today, 0.03, Actual365Fixed()));
    VanillaOption option(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
        ext::make_shared<AmericanExercise>(today, today + Period(12, Months)));
    option.setPricingEngine(
        ext::make_shared<AnalyticDriftedCEVEngine>(100.0, 0.3, 0.5, rTS));
    BOOST_CHECK_THROW(option.NPV(), Error);
}

BOOST_AUTO_TEST_CASE(testFdmDriftedCevEngineArguments) {
    BOOST_TEST_MESSAGE("Testing the mesh and scheme arguments of the drifted FDM CEV engine...");

    // each of the mesh width, the scaling factor and the scheme moves the
    // price by more than the thresholds below (about 2e-3, 9e-4 and 3e-2),
    // so the arguments must reach the solver
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, beta = 0.5, alpha = 0.30*std::pow(f0, 1.0-beta);
    const Handle<YieldTermStructure> rTS(flatRate(today, 0.05, dc));
    VanillaOption option(
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
        ext::make_shared<EuropeanExercise>(today + Period(12, Months)));

    const auto price = [&](Real eps, Real scaling, const FdmSchemeDesc& scheme) -> Real {
        option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
            f0, alpha, beta, rTS, 50, 400, 2, scaling, eps, scheme));
        return option.NPV();
    };

    const Real reference = price(1e-4, 1.0, FdmSchemeDesc::Douglas());
    BOOST_CHECK_GT(std::fabs(price(0.2, 1.0, FdmSchemeDesc::Douglas()) - reference), 5e-4);
    BOOST_CHECK_GT(std::fabs(price(1e-4, 0.7, FdmSchemeDesc::Douglas()) - reference), 2e-4);
    BOOST_CHECK_GT(
        std::fabs(price(1e-4, 1.0, FdmSchemeDesc::ImplicitEuler()) - reference), 5e-3);
}

BOOST_AUTO_TEST_CASE(testFdmDriftedCevGreeksAndGrid) {
    BOOST_TEST_MESSAGE("Testing the Greeks and grid sizes of the drifted FDM CEV engine...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, beta = 0.7, alpha = 0.30*std::pow(f0, 1.0-beta);
    const Handle<YieldTermStructure> rTS(flatRate(today, 0.05, dc));
    const ext::shared_ptr<PlainVanillaPayoff> payoff =
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);

    // reference Greeks by bumping the analytic engine
    const auto analytic = [&](Real f, Integer days) -> Real {
        VanillaOption o(payoff, ext::make_shared<EuropeanExercise>(today + days));
        o.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
            f, alpha, beta, rTS));
        return o.NPV();
    };
    const Real h = 1.0;
    const Real delta = (analytic(f0+h, 365) - analytic(f0-h, 365))/(2*h);
    const Real gamma =
        (analytic(f0+h, 365) - 2*analytic(f0, 365) + analytic(f0-h, 365))/(h*h);
    const Real theta = -(analytic(f0, 370) - analytic(f0, 360))/(10.0/365.0);

    VanillaOption option(payoff, ext::make_shared<EuropeanExercise>(today + 365));
    option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
        f0, alpha, beta, rTS, 100, 400, 2));
    QL_CHECK_SMALL(option.delta() - delta, 1e-3);
    QL_CHECK_SMALL(option.gamma() - gamma, 2e-4);
    QL_CHECK_SMALL(option.theta() - theta, 2e-2);

    // the grid sizes reach the solver: coarse grids move the price
    const auto price = [&](Size tGrid, Size xGrid) -> Real {
        option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
            f0, alpha, beta, rTS, tGrid, xGrid, 2));
        return option.NPV();
    };
    const Real reference = price(50, 400);
    BOOST_CHECK_GT(std::fabs(price(5, 400) - reference), 1e-3);
    BOOST_CHECK_GT(std::fabs(price(50, 25) - reference), 1e-3);
}

BOOST_AUTO_TEST_CASE(testFdmDriftedCevAmericanPut) {
    BOOST_TEST_MESSAGE("Testing early exercise with the drifted FDM CEV engine...");

    // a deep in-the-money put at a high rate is worth more exercised early
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, beta = 0.5, alpha = 3.0, strike = 110.0;
    const Handle<YieldTermStructure> rTS(flatRate(today, 0.08, dc));
    const ext::shared_ptr<PlainVanillaPayoff> payoff =
        ext::make_shared<PlainVanillaPayoff>(Option::Put, strike);
    const Date maturity = today + Period(12, Months);

    VanillaOption european(payoff, ext::make_shared<EuropeanExercise>(maturity));
    VanillaOption american(
        payoff, ext::make_shared<AmericanExercise>(today, maturity));
    const ext::shared_ptr<PricingEngine> engine =
        ext::make_shared<FdDriftedCEVVanillaEngine>(
            f0, alpha, beta, rTS, 200, 400, 2);
    european.setPricingEngine(engine);
    american.setPricingEngine(engine);

    BOOST_CHECK_GT(american.NPV() - european.NPV(), 1.0);
    BOOST_CHECK_GE(american.NPV(), strike - f0);
}

BOOST_AUTO_TEST_CASE(testDriftedEnginesWithLaterReferenceDate) {
    BOOST_TEST_MESSAGE(
        "Testing the drifted CEV engines on a curve that starts later...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    // maturities are measured from the curve's reference date
    const Handle<YieldTermStructure> rTS(
        ext::make_shared<FlatForward>(today + 10, 0.05, dc));
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + 375);
    const Real f0 = 100.0;

    for (Real beta : std::vector<Real>{0.5, 1.5}) {
        const Real alpha = 0.30*std::pow(f0, 1.0-beta);
        VanillaOption option(
            ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0), exercise);
        option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
            f0, alpha, beta, rTS));
        const Real analytic = option.NPV();
        option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
            f0, alpha, beta, rTS, 100, 400, 2));
        QL_CHECK_SMALL(option.NPV() - analytic, 1e-2);
    }
}

BOOST_AUTO_TEST_CASE(testAnalyticCevEngineBetaOne) {
    BOOST_TEST_MESSAGE(
            "Testing analytic CEV engine at beta=1 (the Black limit)...");

    // Regression test: at beta=1, dF = alpha*F*dW is exactly geometric
    // Brownian motion with volatility alpha, so the CEV price must equal
    // the Black formula with stdDev = alpha*sqrt(t). The CEV closed form's
    // own delta_ = (1-2*beta)/(1-beta) has a removable singularity at
    // beta=1 (and is numerically unstable arbitrarily close to it) that
    // the raw formula does not special-case, throwing a raw exception
    // from deep inside the noncentral chi-squared evaluation instead of
    // either pricing correctly or rejecting cleanly.

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Date maturityDate = today + Period(12, Months);
    const Real strike = 2.3;
    const Real f0 = 2.1;
    const Real alpha = 0.75;

    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(maturityDate);
    const ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, 0.15, dc);
    const Time t = dc.yearFraction(today, maturityDate);
    const DiscountFactor discount = rTS->discount(maturityDate);

    for (Real beta : {1.0, 1.0 - 1e-6, 1.0 + 1e-6}) {
        for (Option::Type optionType : {Option::Call, Option::Put}) {
            const ext::shared_ptr<PlainVanillaPayoff> payoff =
                ext::make_shared<PlainVanillaPayoff>(optionType, strike);
            VanillaOption option(payoff, exercise);
            option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
                f0, alpha, beta, Handle<YieldTermStructure>(rTS)));

            Real calculatedNPV = option.NPV();
            Real expectedNPV = discount *
                blackFormula(optionType, strike, f0, alpha*std::sqrt(t));

            if (!std::isfinite(calculatedNPV)) {
                BOOST_ERROR("AnalyticCEVEngine produced a non-finite price "
                            "at beta=" << beta);
            } else if (std::fabs(calculatedNPV - expectedNPV) > 1e-6) {
                BOOST_ERROR("AnalyticCEVEngine at beta=" << beta
                            << " should match the Black price: expected "
                            << expectedNPV << ", got " << calculatedNPV);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testCevEffectiveTimeInputGuards) {
    BOOST_TEST_MESSAGE("Testing the CEV effective time input guards...");

    // a missing discount curve is rejected
    BOOST_CHECK_THROW(
        cevEffectiveTime(0.3, 0.5, ext::shared_ptr<YieldTermStructure>(), 0.0, 1.0),
        Error);
}

BOOST_AUTO_TEST_CASE(testDriftedEnginesAtZeroRate) {
    BOOST_TEST_MESSAGE(
        "Testing the drifted CEV engines against the forward engines at zero rates...");

    // the forward engines price a driftless forward and discount it; the
    // drifted engines price an asset with drift given by the curve. The two
    // conventions coincide at zero rates and differ at positive rates.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, alpha = 0.30*std::pow(100.0, 0.2);
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + Period(12, Months));
    const ext::shared_ptr<PlainVanillaPayoff> payoff =
        ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0);
    VanillaOption option(payoff, exercise);

    const Handle<YieldTermStructure> zero(flatRate(today, 0.0, dc));
    option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
        f0, alpha, 0.8, zero));
    const Real forwardNpv = option.NPV();
    option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
        f0, alpha, 0.8, zero));
    QL_CHECK_SMALL(option.NPV() - forwardNpv, 1e-9);

    option.setPricingEngine(ext::make_shared<FdCEVVanillaEngine>(
        f0, alpha, 0.8, zero, 100, 400, 1));
    const Real fdForwardNpv = option.NPV();
    option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
        f0, alpha, 0.8, zero, 100, 400, 1));
    QL_CHECK_SMALL(option.NPV() - fdForwardNpv, 1e-3);

    const Handle<YieldTermStructure> positive(flatRate(today, 0.03, dc));
    option.setPricingEngine(ext::make_shared<AnalyticCEVEngine>(
        f0, alpha, 0.8, positive));
    const Real forward = option.NPV();
    option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
        f0, alpha, 0.8, positive));
    BOOST_CHECK_GT(option.NPV() - forward, 1.0);
}

BOOST_AUTO_TEST_CASE(testFdmCevOpDirection) {
    BOOST_TEST_MESSAGE("Testing the FDM CEV operator on a non-zero direction...");

    // On a two-dimensional mesh whose second axis is the CEV axis, the
    // operator acting along that axis must reproduce the one-dimensional
    // operator on every row, away from the boundary rows.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, 0.05, dc);
    const Real f0 = 30.0, beta = 0.6, alpha = 0.3*std::pow(f0, 1.0-beta);
    const Size cevSize = 40, otherSize = 4;

    const ext::shared_ptr<Fdm1dMesher> cev =
        ext::make_shared<FdmCEV1dMesher>(cevSize, f0, alpha, beta, 1.0, 1e-3);
    const ext::shared_ptr<FdmMesher> mesher1d =
        ext::make_shared<FdmMesherComposite>(cev);
    const ext::shared_ptr<FdmMesher> mesher2d =
        ext::make_shared<FdmMesherComposite>(
            ext::make_shared<Uniform1dMesher>(0.0, 1.0, otherSize), cev);

    FdmDriftedCEVOp op1d(mesher1d, rTS, f0, alpha, beta, 0);
    FdmDriftedCEVOp op2d(mesher2d, rTS, f0, alpha, beta, 1);
    op1d.setTime(0.0, 0.1);
    op2d.setTime(0.0, 0.1);

    Array g(cevSize);
    for (Size j = 0; j < cevSize; ++j)
        g[j] = std::sin(0.1*cev->location(j)) + 0.01*cev->location(j);
    Array v(cevSize*otherSize);
    const ext::shared_ptr<FdmLinearOpLayout> layout = mesher2d->layout();
    for (Size i = 0; i < otherSize; ++i)
        for (Size j = 0; j < cevSize; ++j)
            v[layout->index(std::vector<Size>{i, j})] = g[j];

    const Array expected = op1d.apply(g);
    const Array calculated = op2d.apply(v);
    // the matrix form is the operator, and there is no mixed term
    const SparseMatrix matrix = op1d.toMatrixDecomp()[0];
    const Array viaMatrix = prod(matrix, g);
    const Array mixed = op1d.apply_mixed(g);
    for (Size j = 0; j < cevSize; ++j) {
        QL_CHECK_SMALL(viaMatrix[j] - expected[j], 1e-10*(1.0 + std::fabs(expected[j])));
        QL_CHECK_SMALL(mixed[j], 1e-14);
    }
    // the splitting along the CEV axis is the whole operator, and the
    // other axis carries no part of it
    const Array direction1 = op2d.apply_direction(1, v);
    const Array direction0 = op2d.apply_direction(0, v);
    for (Size k = 0; k < v.size(); ++k) {
        QL_CHECK_SMALL(direction1[k] - calculated[k], 1e-10*(1.0 + std::fabs(calculated[k])));
        QL_CHECK_SMALL(direction0[k], 1e-14);
    }
    // the implicit solve along the CEV axis acts on every line like the
    // one-dimensional solve
    const Real step = 0.01;
    const Array solved1d = op1d.solve_splitting(0, g, step);
    const Array solved1 = op2d.solve_splitting(1, v, step);
    // and it inverts (I + step*A), with A the operator along that axis
    const Array residual = solved1 + step*op2d.apply_direction(1, solved1);
    const Array preconditioned = op2d.preconditioner(v, step);
    for (Size i = 0; i < otherSize; ++i) {
        for (Size j = 0; j < cevSize; ++j) {
            const Size k = layout->index(std::vector<Size>{i, j});
            QL_CHECK_SMALL(solved1[k] - solved1d[j], 1e-9*(1.0 + std::fabs(solved1d[j])));
            QL_CHECK_SMALL(residual[k] - v[k], 1e-10*(1.0 + std::fabs(v[k])));
            QL_CHECK_SMALL(preconditioned[k] - solved1[k], 1e-12*(1.0 + std::fabs(solved1[k])));
        }
    }
    for (Size i = 0; i < otherSize; ++i) {
        for (Size j = 1; j + 1 < cevSize; ++j) {
            const Real c = calculated[layout->index(std::vector<Size>{i, j})];
            QL_CHECK_SMALL(c - expected[j], 1e-10*(1.0 + std::fabs(expected[j])));
        }
    }
}

BOOST_AUTO_TEST_CASE(testFdmCevOneDayOption) {
    BOOST_TEST_MESSAGE("Testing the FDM CEV engine for a one-day option...");

    // A one-day maturity is shorter than the window floor of the boundary
    // evaluation; with a year fraction that differs from the calendar
    // count the window must stay inside the time to maturity.
    const Date today = Date(15, June, 2020);
    const DayCounter dc = ActualActual(ActualActual::ISDA);
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0;
    const Handle<YieldTermStructure> rTS(flatRate(today, 0.03, dc));
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + 1);

    for (Real beta : std::vector<Real>{0.7, 1.4}) {
        const Real alpha = 0.3*std::pow(f0, 1.0-beta);
        VanillaOption option(
            ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0), exercise);
        option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
            f0, alpha, beta, rTS));
        const Real analytic = option.NPV();
        option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
            f0, alpha, beta, rTS));
        Real pde = Null<Real>();
        BOOST_CHECK_NO_THROW(pde = option.NPV());
        QL_CHECK_SMALL(pde - analytic, 1e-2);
    }
}

BOOST_AUTO_TEST_CASE(testCevBlackBandWidth) {
    BOOST_TEST_MESSAGE("Testing the width of the CEV Black band...");

    // Just outside the band the engine uses the closed form, which differs
    // from the Black price at the local volatility by the skew of the model
    // away from the money (about 6e-4 here); a wider band would return the
    // Black price exactly.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, alpha = 0.30, strike = 130.0;
    const Handle<YieldTermStructure> rTS(flatRate(today, 0.03, dc));
    const Date maturityDate = today + Period(12, Months);
    const auto payoff = ext::make_shared<PlainVanillaPayoff>(Option::Call, strike);
    VanillaOption option(payoff, ext::make_shared<EuropeanExercise>(maturityDate));

    const DiscountFactor discount = rTS->discount(maturityDate);
    const Real beta = 1.0-5e-4;
    const Time maturity = rTS->timeFromReference(maturityDate);
    const Real black = blackFormula(
        payoff, f0/discount,
        std::pow(f0, beta-1.0)
            *std::sqrt(cevEffectiveTime(alpha, beta, rTS.currentLink(), 0.0, maturity)),
        discount);

    option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
        f0, alpha, beta, rTS));
    const Real difference = std::fabs(option.NPV() - black);
    BOOST_CHECK_GT(difference, 1e-4);
    BOOST_CHECK_LT(difference, 2e-3);

    // closer to the band the skew is smaller but still resolved: the band
    // is no wider than 1.5e-4
    option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
        f0, alpha, 1.0-1.5e-4, rTS));
    const Real localBlack = blackFormula(
        payoff, f0/discount,
        std::pow(f0, -1.5e-4)
            *std::sqrt(cevEffectiveTime(
                alpha, 1.0-1.5e-4, rTS.currentLink(), 0.0, maturity)),
        discount);
    BOOST_CHECK_GT(std::fabs(option.NPV() - localBlack), 5e-5);
}

BOOST_AUTO_TEST_CASE(testFdmCevBetaAboveOne) {
    BOOST_TEST_MESSAGE("Testing the FDM CEV engine for beta above one...");

    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    const Real f0 = 100.0, rate = 0.05;
    const Handle<YieldTermStructure> rTS(flatRate(today, rate, dc));
    const ext::shared_ptr<Exercise> exercise =
        ext::make_shared<EuropeanExercise>(today + Period(12, Months));

    // a wide mesh (eps = 0.1) brings the lower boundary close enough for
    // its treatment to matter: beta > 1 has no absorbing boundary at zero.
    for (Real beta : std::vector<Real>{1.5, 2.5}) {
        const Real alpha = 0.30*std::pow(f0, 1.0-beta);
        for (Option::Type type : {Option::Call, Option::Put}) {
            VanillaOption option(
                ext::make_shared<PlainVanillaPayoff>(type, 100.0), exercise);
            option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
                f0, alpha, beta, rTS));
            const Real analytic = option.NPV();
            option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
                f0, alpha, beta, rTS, 200, 400, 1, 1.0, 0.1));
            const Real pde = option.NPV();
            if (std::fabs(analytic - pde) > 0.02) {
                BOOST_ERROR("failed to reproduce the analytic price"
                            << "\n    beta         : " << beta
                            << "\n    analytic npv : " << analytic
                            << "\n    pde npv      : " << pde);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testFdmCevDriftStability) {
    BOOST_TEST_MESSAGE(
        "Testing the FDM CEV engine on long maturities and low volatilities "
        "with a nonzero rate...");

    // Regression test: the drift term must not destabilize the default
    // time grid, neither at the upper boundary (long maturities, where the
    // call value there is large) nor at the lower one (low volatilities,
    // where the mesh starts far above zero), for either sign of the rate.
    const Date today = Date(22, February, 2018);
    const DayCounter dc = Actual365Fixed();
    Settings::instance().evaluationDate() = today;

    struct Case {
        Option::Type type; Real strike; Real vol; Integer days; Rate rate;
        Real beta; Real tolerance;
    };
    const Case cases[] = {
        {Option::Call, 100.0, 0.30, 3650, 0.10, 1.5, 0.05},
        {Option::Call, 100.0, 0.30, 10950, 0.05, 0.5, 0.1},
        {Option::Call, 100.0, 0.30, 1825, 0.20, 1.5, 0.05},
        {Option::Call, 100.0, 0.30, 10950, -0.02, 1.5, 0.1},
        {Option::Call, 90.0, 0.03, 365, 0.10, 0.5, 0.05},
        {Option::Call, 90.0, 0.03, 365, -0.05, 0.5, 0.05},
        {Option::Put, 90.0, 0.03, 1825, -0.05, 0.5, 0.05},
        // beta above one has no absorbing zero, but the mesh still starts
        // above zero for a low volatility
        {Option::Call, 80.0, 0.03, 1825, -0.05, 2.5, 0.05},
        {Option::Put, 80.0, 0.03, 1825, -0.05, 2.5, 0.05},
        {Option::Call, 80.0, 0.03, 1825, -0.05, 1.5, 0.05},
        // close to beta = 1 over a short maturity, where the boundary
        // closed form is near its failure region
        {Option::Call, 100.0, 0.03, 30, 0.0, 0.99, 0.01},
        {Option::Call, 100.0, 0.03, 30, 0.05, 0.99, 0.01}
    };

    const Real f0 = 100.0;
    for (const auto& c : cases) {
        const Handle<YieldTermStructure> rTS(flatRate(today, c.rate, dc));
        const Real alpha = c.vol*std::pow(f0, 1.0-c.beta);
        VanillaOption option(
            ext::make_shared<PlainVanillaPayoff>(c.type, c.strike),
            ext::make_shared<EuropeanExercise>(today + c.days));
        option.setPricingEngine(ext::make_shared<AnalyticDriftedCEVEngine>(
            f0, alpha, c.beta, rTS));
        const Real analytic = option.NPV();
        option.setPricingEngine(ext::make_shared<FdDriftedCEVVanillaEngine>(
            f0, alpha, c.beta, rTS));
        const Real pde = option.NPV();
        if (!(std::fabs(analytic - pde) <= c.tolerance)) {
            BOOST_ERROR("failed to reproduce the analytic price"
                        << "\n    days         : " << c.days
                        << "\n    volatility   : " << c.vol
                        << "\n    rate         : " << c.rate
                        << "\n    beta         : " << c.beta
                        << "\n    analytic npv : " << analytic
                        << "\n    pde npv      : " << pde);
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
