/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2004, 2007, 2008 StatPro Italia srl

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
#include <ql/instruments/cliquetoption.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/integrals/kronrodintegral.hpp>
#include <ql/pricingengines/cliquet/analyticcliquetengine.hpp>
#include <ql/pricingengines/cliquet/analyticperformanceengine.hpp>
#include <ql/pricingengines/cliquet/mcperformanceengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvoltermstructure.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/forwardcurve.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/time/period.hpp>
#include <ql/utilities/dataformatters.hpp>
#include <algorithm>
#include <map>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(CliquetOptionTests)

#undef REPORT_FAILURE
#define REPORT_FAILURE(greekName, payoff, exercise, s, q, r, today, v, \
                       expected, calculated, error, tolerance) \
    BOOST_ERROR(payoff->optionType() << " option:\n" \
               << "    spot value:       " << s << "\n" \
               << "    moneyness:        " << payoff->strike() << "\n" \
               << "    dividend yield:   " << io::rate(q) << "\n" \
               << "    risk-free rate:   " << io::rate(r) << "\n" \
               << "    reference date:   " << today << "\n" \
               << "    maturity:         " << exercise->lastDate() << "\n" \
               << "    volatility:       " << io::volatility(v) << "\n\n" \
               << "    expected   " << greekName << ": " << expected << "\n" \
               << "    calculated " << greekName << ": " << calculated << "\n"\
               << "    error:            " << error << "\n" \
               << "    tolerance:        " << tolerance);

BOOST_AUTO_TEST_CASE(testValues) {

    BOOST_TEST_MESSAGE("Testing Cliquet option values...");

    Date today = Date::todaysDate();
    DayCounter dc = Actual360();

    ext::shared_ptr<SimpleQuote> spot(new SimpleQuote(60.0));
    ext::shared_ptr<SimpleQuote> qRate(new SimpleQuote(0.04));
    ext::shared_ptr<YieldTermStructure> qTS = flatRate(today, qRate, dc);
    ext::shared_ptr<SimpleQuote> rRate(new SimpleQuote(0.08));
    ext::shared_ptr<YieldTermStructure> rTS = flatRate(today, rRate, dc);
    ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(0.30));
    ext::shared_ptr<BlackVolTermStructure> volTS = flatVol(today, vol, dc);

    ext::shared_ptr<BlackScholesMertonProcess> process(
         new BlackScholesMertonProcess(Handle<Quote>(spot),
                                       Handle<YieldTermStructure>(qTS),
                                       Handle<YieldTermStructure>(rTS),
                                       Handle<BlackVolTermStructure>(volTS)));
    ext::shared_ptr<PricingEngine> engine(new AnalyticCliquetEngine(process));

    std::vector<Date> reset;
    reset.push_back(today + 90);
    Date maturity = today + 360;
    Option::Type type = Option::Call;
    Real moneyness = 1.1;

    ext::shared_ptr<PercentageStrikePayoff> payoff(
                                 new PercentageStrikePayoff(type, moneyness));
    ext::shared_ptr<EuropeanExercise> exercise(
                                              new EuropeanExercise(maturity));

    CliquetOption option(payoff, exercise, reset);
    option.setPricingEngine(engine);

    Real calculated = option.NPV();
    Real expected = 4.4064; // Haug, p.37
    Real error = std::fabs(calculated-expected);
    Real tolerance = 1e-4;
    if (error > tolerance) {
        REPORT_FAILURE("value", payoff, exercise, spot->value(),
                       qRate->value(), rRate->value(), today,
                       vol->value(), expected, calculated,
                       error, tolerance);
    }
}


template <class T>
void testOptionGreeks(Real localFloor = Null<Real>(), Real localCap = Null<Real>()) {

    std::map<std::string,Real> calculated, expected, tolerance;
    tolerance["delta"]  = 1.0e-5;
    tolerance["gamma"]  = 1.0e-5;
    tolerance["theta"]  = 1.0e-5;
    tolerance["rho"]    = 1.0e-5;
    tolerance["divRho"] = 1.0e-5;
    tolerance["vega"]   = 1.0e-5;

    Option::Type types[] = { Option::Call, Option::Put };
    Real moneyness[] = { 0.9, 1.0, 1.1 };
    Real underlyings[] = { 100.0 };
    Rate qRates[] = { 0.04, 0.05, 0.06 };
    Rate rRates[] = { 0.01, 0.05, 0.15 };
    Integer lengths[] = { 1, 2 };
    Frequency frequencies[] = { Semiannual, Quarterly };
    Volatility vols[] = { 0.11, 0.50, 1.20 };

    DayCounter dc = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;

    ext::shared_ptr<SimpleQuote> spot(new SimpleQuote(0.0));
    ext::shared_ptr<SimpleQuote> qRate(new SimpleQuote(0.0));
    Handle<YieldTermStructure> qTS(flatRate(qRate, dc));
    ext::shared_ptr<SimpleQuote> rRate(new SimpleQuote(0.0));
    Handle<YieldTermStructure> rTS(flatRate(rRate, dc));
    ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(0.0));
    Handle<BlackVolTermStructure> volTS(flatVol(vol, dc));

    ext::shared_ptr<BlackScholesMertonProcess> process(
                            new BlackScholesMertonProcess(Handle<Quote>(spot),
                                                          qTS, rTS, volTS));

    for (auto& type : types) {
        for (Real moneynes : moneyness) {
            for (int length : lengths) {
                for (auto& frequencie : frequencies) {

                    ext::shared_ptr<EuropeanExercise> maturity(
                            new EuropeanExercise(today + length * Years));

                    ext::shared_ptr<PercentageStrikePayoff> payoff(
                            new PercentageStrikePayoff(type, moneynes));

                    std::vector<Date> reset;
                    for (Date d = today + Period(frequencie); d < maturity->lastDate();
                         d += Period(frequencie))
                        reset.push_back(d);

                    ext::shared_ptr<PricingEngine> engine(new T(process));

                    CliquetOption option(payoff, maturity, reset, localFloor, localCap);
                    option.setPricingEngine(engine);

                    for (Real u : underlyings) {
                        for (Real m : qRates) {
                            for (Real n : rRates) {
                                for (Real v : vols) {

                                    Rate q = m, r = n;
                                    spot->setValue(u);
                                    qRate->setValue(q);
                                    rRate->setValue(r);
                                    vol->setValue(v);

                                    Real value = option.NPV();
                                    calculated["delta"] = option.delta();
                                    calculated["gamma"] = option.gamma();
                                    calculated["theta"] = option.theta();
                                    calculated["rho"] = option.rho();
                                    calculated["divRho"] = option.dividendRho();
                                    calculated["vega"] = option.vega();

                                    if (value > spot->value() * 1.0e-5) {
                                        // perturb spot and get delta and gamma
                                        Real du = u * 1.0e-4;
                                        spot->setValue(u + du);
                                        Real value_p = option.NPV(), delta_p = option.delta();
                                        spot->setValue(u - du);
                                        Real value_m = option.NPV(), delta_m = option.delta();
                                        spot->setValue(u);
                                        expected["delta"] = (value_p - value_m) / (2 * du);
                                        expected["gamma"] = (delta_p - delta_m) / (2 * du);

                                        // perturb rates and get rho and dividend rho
                                        Spread dr = r * 1.0e-4;
                                        rRate->setValue(r + dr);
                                        value_p = option.NPV();
                                        rRate->setValue(r - dr);
                                        value_m = option.NPV();
                                        rRate->setValue(r);
                                        expected["rho"] = (value_p - value_m) / (2 * dr);

                                        Spread dq = q * 1.0e-4;
                                        qRate->setValue(q + dq);
                                        value_p = option.NPV();
                                        qRate->setValue(q - dq);
                                        value_m = option.NPV();
                                        qRate->setValue(q);
                                        expected["divRho"] = (value_p - value_m) / (2 * dq);

                                        // perturb volatility and get vega
                                        Volatility dv = v * 1.0e-4;
                                        vol->setValue(v + dv);
                                        value_p = option.NPV();
                                        vol->setValue(v - dv);
                                        value_m = option.NPV();
                                        vol->setValue(v);
                                        expected["vega"] = (value_p - value_m) / (2 * dv);

                                        // perturb date and get theta
                                        Time dT = dc.yearFraction(today - 1, today + 1);
                                        Settings::instance().evaluationDate() = today - 1;
                                        value_m = option.NPV();
                                        Settings::instance().evaluationDate() = today + 1;
                                        value_p = option.NPV();
                                        Settings::instance().evaluationDate() = today;
                                        expected["theta"] = (value_p - value_m) / dT;

                                        // compare
                                        std::map<std::string, Real>::iterator it;
                                        for (it = calculated.begin(); it != calculated.end();
                                             ++it) {
                                            std::string greek = it->first;
                                            Real expct = expected[greek],
                                                calcl = calculated[greek],
                                                tol = tolerance[greek];
                                            Real error = relativeError(expct, calcl, u);
                                            if (error > tol) {
                                                REPORT_FAILURE(greek, payoff, maturity, u, q, r,
                                                               today, v, expct, calcl, error,
                                                               tol);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


BOOST_AUTO_TEST_CASE(testGreeks) {
    BOOST_TEST_MESSAGE("Testing Cliquet option greeks...");
    testOptionGreeks<AnalyticCliquetEngine>();
}

BOOST_AUTO_TEST_CASE(testPerformanceGreeks) {
    BOOST_TEST_MESSAGE("Testing performance option greeks...");
    testOptionGreeks<AnalyticPerformanceEngine>();
}

BOOST_AUTO_TEST_CASE(testLocalFloorCapGreeks) {
    BOOST_TEST_MESSAGE("Testing cliquet and performance option greeks with local floors "
                       "and caps...");
    const std::pair<Real, Real> bounds[] = {
        {0.02, Null<Real>()}, {Null<Real>(), 0.05}, {0.02, 0.08}};
    for (const auto& bound : bounds) {
        testOptionGreeks<AnalyticCliquetEngine>(bound.first, bound.second);
        testOptionGreeks<AnalyticPerformanceEngine>(bound.first, bound.second);
    }
}

// A forward curve that follows the given forward rates between the given dates, whatever its
// reference date: the rates are fixed in calendar time and the first one applies from the
// reference date to the first node.
ext::shared_ptr<YieldTermStructure> calendarForwardCurve(const Date& referenceDate,
                                                         const std::vector<Date>& nodes,
                                                         const std::vector<Rate>& forwards,
                                                         const DayCounter& dc) {
    std::vector<Date> dates(1, referenceDate);
    dates.insert(dates.end(), nodes.begin(), nodes.end());
    std::vector<Rate> rates(1, forwards.front());
    rates.insert(rates.end(), forwards.begin(), forwards.end());
    return ext::make_shared<ForwardCurve>(dates, rates, dc);
}

// Theta against the finite-difference derivative of the engine's own price when the evaluation
// date, and with it the reference date of the curves, moves by one day on each side.
template <class T>
void testThetaOnForwardCurves(const std::vector<Rate>& qForwards,
                              const std::vector<Rate>& rForwards,
                              const DayCounter& divdc,
                              const DayCounter& rfdc,
                              Real localFloor = Null<Real>(),
                              Real localCap = Null<Real>()) {

    Date today = Settings::instance().evaluationDate();
    std::vector<Date> nodes = {today + 30, today + 120, today + 240, today + 400};
    std::vector<Date> reset = {today + 90, today + 180, today + 270};
    auto exercise = ext::make_shared<EuropeanExercise>(today + 360);

    auto makeEngine = [&](const Date& referenceDate) {
        Handle<Quote> spot(ext::make_shared<SimpleQuote>(100.0));
        Handle<YieldTermStructure> qTS(
            calendarForwardCurve(referenceDate, nodes, qForwards, divdc));
        Handle<YieldTermStructure> rTS(calendarForwardCurve(referenceDate, nodes, rForwards, rfdc));
        Handle<BlackVolTermStructure> volTS(
            ext::make_shared<BlackConstantVol>(referenceDate, NullCalendar(), 0.25, rfdc));
        return ext::make_shared<T>(
            ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS));
    };

    Option::Type types[] = {Option::Call, Option::Put};
    for (auto& type : types) {
        BOOST_TEST_CONTEXT(type << " option") {
            auto payoff = ext::make_shared<PercentageStrikePayoff>(type, 1.05);
            CliquetOption option(payoff, exercise, reset, localFloor, localCap);

            option.setPricingEngine(makeEngine(today));
            Real calculated = option.theta();

            Settings::instance().evaluationDate() = today - 1;
            option.setPricingEngine(makeEngine(today - 1));
            Real value_m = option.NPV();
            Settings::instance().evaluationDate() = today + 1;
            option.setPricingEngine(makeEngine(today + 1));
            Real value_p = option.NPV();
            Settings::instance().evaluationDate() = today;

            Real expected = (value_p - value_m) / rfdc.yearFraction(today - 1, today + 1);
            QL_CHECK_CLOSE_FRACTION(calculated, expected, 1.0e-7);
        }
    }
}

BOOST_AUTO_TEST_CASE(testThetaOnNonFlatCurves) {
    BOOST_TEST_MESSAGE("Testing cliquet and performance option theta on non-flat curves...");

    // forward rates up to 30, 120, 240 and 400 days after the evaluation date; the first reset
    // date is 90 days after it
    const std::vector<Rate> flatQ(4, 0.02), flatR(4, 0.05);
    const std::vector<Rate> varyingQ = {0.01, 0.04, 0.015, 0.06};
    const std::vector<Rate> varyingR = {0.02, 0.05, 0.08, 0.03};

    // theta is in years of the risk-free curve's day counter, also when the dividend curve
    // has another one
    const DayCounter divdc = Actual365Fixed();
    const DayCounter rfdcs[] = {Actual365Fixed(), Actual360()};

    auto checkTheta = [&](const char* curves, const std::vector<Rate>& q,
                          const std::vector<Rate>& r) {
        for (const DayCounter& rfdc : rfdcs) {
            BOOST_TEST_CONTEXT(curves << ", risk-free day counter " << rfdc.name()) {
                BOOST_TEST_CONTEXT("AnalyticCliquetEngine") {
                    testThetaOnForwardCurves<AnalyticCliquetEngine>(q, r, divdc, rfdc);
                }
                BOOST_TEST_CONTEXT("AnalyticPerformanceEngine") {
                    testThetaOnForwardCurves<AnalyticPerformanceEngine>(q, r, divdc, rfdc);
                }
                BOOST_TEST_CONTEXT("AnalyticCliquetEngine, local floor and cap") {
                    testThetaOnForwardCurves<AnalyticCliquetEngine>(q, r, divdc, rfdc, 0.02, 0.08);
                }
                BOOST_TEST_CONTEXT("AnalyticPerformanceEngine, local floor and cap") {
                    testThetaOnForwardCurves<AnalyticPerformanceEngine>(q, r, divdc, rfdc, 0.02,
                                                                        0.08);
                }
            }
        }
    };

    // theta on flat curves does not change
    checkTheta("flat curves", flatQ, flatR);
    // the rates before the first reset date differ from those after it
    checkTheta("non-flat curves", varyingQ, varyingR);
}

BOOST_AUTO_TEST_CASE(testMcPerformance) {
    BOOST_TEST_MESSAGE(
        "Testing Monte Carlo performance engine against analytic results...");

    Option::Type types[] = { Option::Call, Option::Put };
    Real moneyness[] = { 0.9, 1.1 };
    Real underlyings[] = { 100.0 };
    Rate qRates[] = { 0.04, 0.06 };
    Rate rRates[] = { 0.01, 0.10 };
    Integer lengths[] = { 2, 4 };
    Frequency frequencies[] = { Semiannual, Quarterly };
    Volatility vols[] = { 0.10, 0.90 };

    DayCounter dc = Actual360();
    Date today = Date::todaysDate();
    Settings::instance().evaluationDate() = today;

    ext::shared_ptr<SimpleQuote> spot(new SimpleQuote(0.0));
    ext::shared_ptr<SimpleQuote> qRate(new SimpleQuote(0.0));
    Handle<YieldTermStructure> qTS(flatRate(qRate, dc));
    ext::shared_ptr<SimpleQuote> rRate(new SimpleQuote(0.0));
    Handle<YieldTermStructure> rTS(flatRate(rRate, dc));
    ext::shared_ptr<SimpleQuote> vol(new SimpleQuote(0.0));
    Handle<BlackVolTermStructure> volTS(flatVol(vol, dc));

    ext::shared_ptr<BlackScholesMertonProcess> process(
                            new BlackScholesMertonProcess(Handle<Quote>(spot),
                                                          qTS, rTS, volTS));

    for (auto& type : types) {
        for (Real moneynes : moneyness) {
            for (int length : lengths) {
                for (auto& frequencie : frequencies) {

                    auto tenor = Period(frequencie);
                    ext::shared_ptr<EuropeanExercise> maturity(
                        new EuropeanExercise(today + length * tenor));

                    ext::shared_ptr<PercentageStrikePayoff> payoff(
                        new PercentageStrikePayoff(type, moneynes));

                    std::vector<Date> reset;
                    for (Date d = today + tenor; d < maturity->lastDate(); d += tenor)
                        reset.push_back(d);

                    CliquetOption option(payoff, maturity, reset);

                    ext::shared_ptr<PricingEngine> refEngine(
                        new AnalyticPerformanceEngine(process));

                    ext::shared_ptr<PricingEngine> mcEngine =
                        MakeMCPerformanceEngine<PseudoRandom>(process)
                            .withBrownianBridge()
                            .withAbsoluteTolerance(5.0e-3)
                            .withSeed(42);

                    for (Real u : underlyings) {
                        for (Real m : qRates) {
                            for (Real n : rRates) {
                                for (Real v : vols) {

                                    Rate q = m, r = n;
                                    spot->setValue(u);
                                    qRate->setValue(q);
                                    rRate->setValue(r);
                                    vol->setValue(v);

                                    option.setPricingEngine(refEngine);
                                    Real refValue = option.NPV();

                                    option.setPricingEngine(mcEngine);
                                    Real value = option.NPV();

                                    Real error = std::fabs(refValue - value);
                                    Real tolerance = 1.5e-2;
                                    if (error > tolerance) {
                                        REPORT_FAILURE("value", payoff, maturity, u, q, r, today, v,
                                                       refValue, value, error, tolerance);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

namespace {

    // the period payoff of a percentage-strike option on the ratio x of the prices at the end and
    // at the start of the period, clamped between the local floor and cap
    Real clampedCoupon(Option::Type type, Real moneyness, Real x, Real floor, Real cap) {
        Real coupon = std::max<Real>(int(type) * (x - moneyness), 0.0);
        if (floor != Null<Real>())
            coupon = std::max<Real>(coupon, floor);
        if (cap != Null<Real>())
            coupon = std::min<Real>(coupon, cap);
        return coupon;
    }

    // Expected value of the clamped coupon of one period under the lognormal law of the ratio,
    // by adaptive integration over the standard normal variable.
    Real expectedCoupon(Option::Type type, Real moneyness, Real mu, Real stdDev,
                        Real floor, Real cap) {
        static const GaussKronrodAdaptive integrator(1.0e-13, 100000);
        static const NormalDistribution gaussian;
        auto f = [&](Real z) -> Real {
            return gaussian(z) *
                   clampedCoupon(type, moneyness, std::exp(mu + stdDev * z), floor, cap);
        };
        // integrate between the kinks of the clamped payoff, which are where the ratio equals
        // the strikes of the unclamped payoff and of the floored and capped levels
        const Real sgn = type == Option::Call ? 1.0 : -1.0;
        std::vector<Real> points = {-12.0, 12.0};
        for (Real bound : std::vector<Real>{Real(0.0), floor == Null<Real>() ? Real(0.0) : floor,
                                            cap == Null<Real>() ? Real(0.0) : cap}) {
            Real strike = moneyness + sgn * bound;
            Real z = strike > 0.0 ? (std::log(strike) - mu) / stdDev : -100.0;
            if (z > -12.0 && z < 12.0)
                points.push_back(z);
        }
        std::sort(points.begin(), points.end());
        Real sum = 0.0;
        for (Size i = 1; i < points.size(); ++i)
            sum += integrator(f, points[i - 1], points[i]);
        return sum;
    }

    struct Bounds {
        Real floor, cap;
    };

    const Real noBound = Null<Real>();
    const Bounds boundCases[] = {{noBound, 0.06}, {0.01, noBound}, {0.0, 0.06},
                                 {0.01, 0.06},    {0.03, 0.03},    {0.02, 1.0}};

    // spot, rates and volatility of the tests below, on flat curves
    struct FlatMarket {
        Real spot = 100.0;
        Rate q = 0.03, r = 0.05;
        Volatility sigma = 0.25;
        DayCounter dc = Actual365Fixed();

        ext::shared_ptr<GeneralizedBlackScholesProcess> process(const Date& today) const {
            return ext::make_shared<BlackScholesMertonProcess>(
                Handle<Quote>(ext::make_shared<SimpleQuote>(spot)),
                Handle<YieldTermStructure>(ext::make_shared<FlatForward>(today, q, dc)),
                Handle<YieldTermStructure>(ext::make_shared<FlatForward>(today, r, dc)),
                Handle<BlackVolTermStructure>(
                    ext::make_shared<BlackConstantVol>(today, NullCalendar(), sigma, dc)));
        }
    };

}

BOOST_AUTO_TEST_CASE(testLocalFloorCapValues) {
    BOOST_TEST_MESSAGE("Testing cliquet and performance option values with local floors "
                       "and caps...");

    Date today = Settings::instance().evaluationDate();
    FlatMarket market;
    auto process = market.process(today);
    std::vector<Date> reset = {today + 90, today + 180, today + 270};
    auto exercise = ext::make_shared<EuropeanExercise>(today + 360);
    std::vector<Date> dates = reset;
    dates.push_back(exercise->lastDate());

    // the forward of the ratio over the first period is about 1.005
    const Real moneyness[] = {0.95, 1.0, 1.005, 1.05};
    for (Option::Type type : {Option::Call, Option::Put}) {
        for (Real k : moneyness) {
            // a cap or floor at the moneyness leaves a put with a strike of zero
            std::vector<Bounds> cases(std::begin(boundCases), std::end(boundCases));
            cases.push_back({noBound, k});
            cases.push_back({k, k});
            for (const Bounds& bounds : cases) {
                BOOST_TEST_CONTEXT(type << ", moneyness " << k << ", floor " << bounds.floor
                                        << ", cap " << bounds.cap) {
                    auto payoff = ext::make_shared<PercentageStrikePayoff>(type, k);
                    CliquetOption option(payoff, exercise, reset, bounds.floor, bounds.cap);

                    // the reference value, by integrating the clamped payoff directly
                    Real expectedCliquet = 0.0, expectedPerformance = 0.0;
                    // the reset dates are the starts of the periods
                    for (Size i = 1; i < dates.size(); ++i) {
                        Time t0 = market.dc.yearFraction(today, dates[i - 1]);
                        Time t1 = market.dc.yearFraction(today, dates[i]);
                        Time dt = t1 - t0;
                        Real mu = (market.r - market.q - 0.5 * market.sigma * market.sigma) * dt;
                        Real e = expectedCoupon(type, k, mu, market.sigma * std::sqrt(dt),
                                                bounds.floor, bounds.cap);
                        expectedPerformance += std::exp(-market.r * t1) * e;
                        expectedCliquet +=
                            market.spot * std::exp(-market.q * t0 - market.r * dt) * e;
                    }

                    option.setPricingEngine(ext::make_shared<AnalyticCliquetEngine>(process));
                    QL_CHECK_CLOSE_FRACTION(option.NPV(), expectedCliquet, 1.0e-9);

                    option.setPricingEngine(ext::make_shared<AnalyticPerformanceEngine>(process));
                    QL_CHECK_CLOSE_FRACTION(option.NPV(), expectedPerformance, 1.0e-9);
                }
            }
        }
    }
}

namespace {

    // A volatility depending on the strike, to tell the strike at which each leg of a clamped
    // payoff reads it.
    class SkewedVol : public BlackVolatilityTermStructure {
      public:
        SkewedVol(const Date& referenceDate, const DayCounter& dc)
        : BlackVolatilityTermStructure(referenceDate, NullCalendar(), Following, dc) {}
        Date maxDate() const override { return Date::maxDate(); }
        Real minStrike() const override { return 0.0; }
        Real maxStrike() const override { return QL_MAX_REAL; }

      protected:
        Volatility blackVolImpl(Time, Real strike) const override {
            return 0.15 + 0.3 * std::exp(-strike / 100.0);
        }
    };

}

BOOST_AUTO_TEST_CASE(testLocalFloorCapOnSmile) {
    BOOST_TEST_MESSAGE("Testing cliquet and performance option with local floors and caps "
                       "on a volatility depending on the strike...");

    Date today = Settings::instance().evaluationDate();
    FlatMarket market;
    Handle<Quote> spot(ext::make_shared<SimpleQuote>(market.spot));
    Handle<YieldTermStructure> qTS(ext::make_shared<FlatForward>(today, market.q, market.dc));
    Handle<YieldTermStructure> rTS(ext::make_shared<FlatForward>(today, market.r, market.dc));
    Handle<BlackVolTermStructure> volTS(ext::make_shared<SkewedVol>(today, market.dc));
    auto process = ext::make_shared<BlackScholesMertonProcess>(spot, qTS, rTS, volTS);
    std::vector<Date> reset = {today + 90, today + 180, today + 270};
    auto exercise = ext::make_shared<EuropeanExercise>(today + 360);

    // the amount received at the end of each period, per unit of floor
    Real unitPerformance = 0.0, unitCliquet = 0.0;
    std::vector<Date> dates = reset;
    dates.push_back(exercise->lastDate());
    for (Size i = 1; i < dates.size(); ++i) {
        unitPerformance += rTS->discount(dates[i]);
        unitCliquet += market.spot * qTS->discount(dates[i - 1]) *
                       rTS->discount(dates[i]) / rTS->discount(dates[i - 1]);
    }

    const Real floor = 0.01, cap = 0.05, k = 1.0;
    for (Option::Type type : {Option::Call, Option::Put}) {
        // the clamped payoff is the floor plus the options struck at the floored and capped levels
        const Real sgn = type == Option::Call ? 1.0 : -1.0;
        auto price = [&](const ext::shared_ptr<PricingEngine>& engine, Real moneyness) -> Real {
            auto payoff = ext::make_shared<PercentageStrikePayoff>(type, moneyness);
            CliquetOption option(payoff, exercise, reset);
            option.setPricingEngine(engine);
            return option.NPV();
        };
        BOOST_TEST_CONTEXT(type << " option") {
            auto cliquet = ext::make_shared<AnalyticCliquetEngine>(process);
            auto performance = ext::make_shared<AnalyticPerformanceEngine>(process);
            auto payoff = ext::make_shared<PercentageStrikePayoff>(type, k);
            CliquetOption option(payoff, exercise, reset, floor, cap);

            option.setPricingEngine(cliquet);
            Real expected = floor * unitCliquet + price(cliquet, k + sgn * floor) -
                            price(cliquet, k + sgn * cap);
            QL_CHECK_CLOSE_FRACTION(option.NPV(), expected, 1.0e-12);

            option.setPricingEngine(performance);
            expected = floor * unitPerformance + price(performance, k + sgn * floor) -
                       price(performance, k + sgn * cap);
            QL_CHECK_CLOSE_FRACTION(option.NPV(), expected, 1.0e-12);
        }
    }
}

BOOST_AUTO_TEST_CASE(testLocalFloorCapDefaults) {
    BOOST_TEST_MESSAGE("Testing that missing local floors and caps leave cliquet and "
                       "performance option values unchanged...");

    Date today = Settings::instance().evaluationDate();
    auto process = FlatMarket().process(today);
    std::vector<Date> reset = {today + 90, today + 180};
    auto exercise = ext::make_shared<EuropeanExercise>(today + 270);

    for (Option::Type type : {Option::Call, Option::Put}) {
        auto payoff = ext::make_shared<PercentageStrikePayoff>(type, 1.02);
        const ext::shared_ptr<PricingEngine> engines[] = {
            ext::make_shared<AnalyticCliquetEngine>(process),
            ext::make_shared<AnalyticPerformanceEngine>(process)};
        for (const auto& engine : engines) {
            CliquetOption plain(payoff, exercise, reset);
            CliquetOption zeroFloor(payoff, exercise, reset, 0.0);
            CliquetOption wideCap(payoff, exercise, reset, Null<Real>(), 1.0e6);
            plain.setPricingEngine(engine);
            zeroFloor.setPricingEngine(engine);
            wideCap.setPricingEngine(engine);
            // a zero floor is the implicit floor of the unclamped payoff
            BOOST_CHECK_EQUAL(value(plain.NPV()), value(zeroFloor.NPV()));
            QL_CHECK_CLOSE_FRACTION(plain.NPV(), wideCap.NPV(), 1.0e-13);
        }
    }
}

BOOST_AUTO_TEST_CASE(testLocalFloorCapMonteCarlo) {
    BOOST_TEST_MESSAGE("Testing Monte Carlo performance engine with local floors "
                       "and caps against analytic results...");

    Date today = Settings::instance().evaluationDate();
    FlatMarket market;
    market.sigma = 0.30;
    auto process = market.process(today);
    std::vector<Date> reset = {today + 90, today + 180, today + 270};
    auto exercise = ext::make_shared<EuropeanExercise>(today + 360);
    auto analytic = ext::make_shared<AnalyticPerformanceEngine>(process);
    ext::shared_ptr<PricingEngine> mc = MakeMCPerformanceEngine<PseudoRandom>(process)
                                            .withBrownianBridge()
                                            .withAbsoluteTolerance(2.0e-4)
                                            .withSeed(42);

    const Bounds bounds[] = {{noBound, 0.04}, {0.01, noBound}, {0.01, 0.05}};
    for (Option::Type type : {Option::Call, Option::Put}) {
        for (Real k : {0.97, 1.0, 1.03}) {
            for (const Bounds& b : bounds) {
                BOOST_TEST_CONTEXT(type << ", moneyness " << k << ", floor " << b.floor
                                        << ", cap " << b.cap) {
                    auto payoff = ext::make_shared<PercentageStrikePayoff>(type, k);
                    CliquetOption option(payoff, exercise, reset, b.floor, b.cap);
                    option.setPricingEngine(analytic);
                    Real expected = option.NPV();
                    option.setPricingEngine(mc);
                    QL_CHECK_SMALL(option.NPV() - expected, 1.0e-3);
                }
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(testUnsupportedBounds) {
    BOOST_TEST_MESSAGE("Testing that the cliquet and performance engines refuse the options "
                       "they cannot price...");

    Date today = Settings::instance().evaluationDate();
    auto process = FlatMarket().process(today);
    std::vector<Date> reset = {today + 90};
    auto exercise = ext::make_shared<EuropeanExercise>(today + 180);
    auto payoff = ext::make_shared<PercentageStrikePayoff>(Option::Call, 1.0);

    // a floor above the cap is rejected whichever engine is used
    CliquetOption inverted(payoff, exercise, reset, 0.05, 0.02);
    inverted.setPricingEngine(ext::make_shared<AnalyticPerformanceEngine>(process));
    BOOST_CHECK_THROW(inverted.NPV(), Error);

    const ext::shared_ptr<PricingEngine> engines[] = {
        ext::make_shared<AnalyticCliquetEngine>(process),
        ext::make_shared<AnalyticPerformanceEngine>(process),
        MakeMCPerformanceEngine<PseudoRandom>(process).withSamples(100).withSeed(42)};
    CliquetOption option(payoff, exercise, reset);
    for (const auto& engine : engines) {
        option.setupArguments(engine->getArguments());
        auto* args = dynamic_cast<CliquetOption::arguments*>(engine->getArguments());
        BOOST_REQUIRE(args != nullptr);
        Real* fields[] = {&args->accruedCoupon, &args->lastFixing, &args->globalCap,
                          &args->globalFloor};
        for (Real* field : fields) {
            *field = 0.5;
            BOOST_CHECK_THROW(engine->calculate(), Error);
            *field = Null<Real>();
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
