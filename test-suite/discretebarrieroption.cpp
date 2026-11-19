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
#include <ql/exercise.hpp>
#include <ql/instruments/barrieroption.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/matrix.hpp>
#include <ql/math/randomnumbers/mt19937uniformrng.hpp>
#include <ql/math/statistics/incrementalstatistics.hpp>
#include <ql/pricingengines/barrier/analyticbarrierengine.hpp>
#include <ql/pricingengines/barrier/analyticdiscretebarrierengine.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancesurface.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <sstream>
#include <utility>
#include <vector>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(DiscreteBarrierOptionTests)

namespace {

    // Maturity defaults to exactly one year on the Actual/365 (Fixed) clock.
    struct CommonVars {
        DayCounter dc = Actual365Fixed();
        Date today = Date(15, November, 2026);
        Date expiry = today + 1 * Years;
        ext::shared_ptr<SimpleQuote> spot = ext::make_shared<SimpleQuote>(100.0);
        ext::shared_ptr<SimpleQuote> rate = ext::make_shared<SimpleQuote>(0.05);
        ext::shared_ptr<SimpleQuote> dividend = ext::make_shared<SimpleQuote>(0.02);
        ext::shared_ptr<SimpleQuote> vol = ext::make_shared<SimpleQuote>(0.25);
        ext::shared_ptr<GeneralizedBlackScholesProcess> process;

        CommonVars() {
            Settings::instance().evaluationDate() = today;
            process = ext::make_shared<BlackScholesMertonProcess>(
                Handle<Quote>(spot),
                Handle<YieldTermStructure>(flatRate(today, dividend, dc)),
                Handle<YieldTermStructure>(flatRate(today, rate, dc)),
                Handle<BlackVolTermStructure>(flatVol(today, vol, dc)));
        }

        BarrierOption option(Barrier::Type type, Real barrier, Option::Type optionType,
                             Real strike = 100.0, Real rebate = 0.0) const {
            return BarrierOption(type, barrier, rebate,
                                 ext::make_shared<PlainVanillaPayoff>(optionType, strike),
                                 ext::make_shared<EuropeanExercise>(expiry));
        }

        Real discrete(Barrier::Type type, Real barrier, Option::Type optionType, Size steps,
                      Real strike = 100.0) const {
            BarrierOption o = option(type, barrier, optionType, strike);
            o.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(process, steps));
            return o.NPV();
        }

        Real continuous(Barrier::Type type, Real barrier, Option::Type optionType,
                        Real strike = 100.0) const {
            BarrierOption o = option(type, barrier, optionType, strike);
            o.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(process));
            return o.NPV();
        }
    };

    // Exact lognormal steps at the N monitoring dates: no time-discretization error.
    // Returns the discounted mean payoff and its standard error.
    std::pair<Real, Real> monteCarlo(const CommonVars& vars, Barrier::Type type, Real barrier,
                                     Option::Type optionType, Size steps, Size paths) {
        Real s0 = vars.spot->value(), strike = 100.0, r = vars.rate->value(),
             q = vars.dividend->value(), sigma = vars.vol->value();
        Real maturity = vars.dc.yearFraction(vars.today, vars.expiry);
        Real dt = maturity / Real(steps);
        Real drift = (r - q - 0.5 * sigma * sigma) * dt, diffusion = sigma * std::sqrt(dt);
        bool down = (type == Barrier::DownIn || type == Barrier::DownOut);
        bool knockIn = (type == Barrier::DownIn || type == Barrier::UpIn);
        MersenneTwisterUniformRng uniform(20261119);
        InverseCumulativeNormal inverse;
        IncrementalStatistics stats;
        for (Size i = 0; i < paths; ++i) {
            Real logS = std::log(s0);
            bool hit = false;
            for (Size j = 0; j < steps; ++j) {
                logS += drift + diffusion * inverse(uniform.next().value);
                Real s = std::exp(logS);
                hit = hit || (down ? s <= barrier : s >= barrier);
            }
            Real st = std::exp(logS);
            Real payoff = (optionType == Option::Call) ? std::max<Real>(st - strike, 0.0)
                                                       : std::max<Real>(strike - st, 0.0);
            stats.add((hit == knockIn) ? Real(std::exp(-r * maturity) * payoff) : Real(0.0));
        }
        return {stats.mean(), stats.errorEstimate()};
    }

}

BOOST_AUTO_TEST_CASE(testReferenceValues) {
    BOOST_TEST_MESSAGE("Testing discrete barrier prices against the shifted-barrier formula...");

    // Reiner-Rubinstein prices at the shifted barrier H exp(+-beta sigma sqrt(T/N)), computed
    // outside QuantLib.  S = 100, K = 100, r = 5%, q = 2%, sigma = 25%, T = 1.
    struct Case {
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Size steps;
        Real expected;
    };
    const Case cases[] = {
        {Barrier::DownOut, 90.0, Option::Call, 12, 9.5740025184},
        {Barrier::DownIn, 90.0, Option::Call, 12, 1.5497594097},
        {Barrier::DownOut, 90.0, Option::Put, 12, 0.2962311188},
        {Barrier::DownIn, 90.0, Option::Put, 12, 7.9306059287},
        {Barrier::UpOut, 120.0, Option::Call, 12, 1.3308536713},
        {Barrier::UpIn, 120.0, Option::Call, 12, 9.7929082568},
        {Barrier::UpOut, 120.0, Option::Put, 12, 7.9103967300},
        {Barrier::UpIn, 120.0, Option::Put, 12, 0.3164403175},
        {Barrier::DownOut, 90.0, Option::Call, 50, 8.9369302463},
        {Barrier::UpIn, 120.0, Option::Call, 50, 10.1605931817},
        {Barrier::DownIn, 90.0, Option::Put, 50, 8.0586098363},
        {Barrier::UpOut, 120.0, Option::Put, 50, 7.7474057970},
    };

    CommonVars vars;
    for (const auto& c : cases) {
        Real calculated = vars.discrete(c.type, c.barrier, c.optionType, c.steps);
        QL_CHECK_SMALL(calculated - c.expected, 1.0e-8);
    }
}

BOOST_AUTO_TEST_CASE(testReferenceValuesOtherMaturities) {
    BOOST_TEST_MESSAGE("Testing discrete barrier prices at maturities other than one year...");

    // As in testReferenceValues, with 181 days (T = 181/365) and 731 days (T = 731/365).
    struct Case {
        Date::serial_type days;
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Real expected;
    };
    const Case cases[] = {
        {181, Barrier::DownOut, 90.0, Option::Call, 7.1506589134},
        {181, Barrier::DownIn, 90.0, Option::Put, 5.6564033695},
        {181, Barrier::UpIn, 120.0, Option::Call, 5.4564736195},
        {181, Barrier::UpOut, 120.0, Option::Put, 6.1438527113},
        {731, Barrier::DownOut, 90.0, Option::Call, 12.5004180490},
        {731, Barrier::DownIn, 90.0, Option::Put, 10.3137504186},
        {731, Barrier::UpIn, 120.0, Option::Call, 15.3470218519},
        {731, Barrier::UpOut, 120.0, Option::Put, 9.3739580301},
    };

    CommonVars vars;
    for (const auto& c : cases) {
        vars.expiry = vars.today + c.days;
        Real calculated = vars.discrete(c.type, c.barrier, c.optionType, 12);
        QL_CHECK_SMALL(calculated - c.expected, 1.0e-8);
    }
}

BOOST_AUTO_TEST_CASE(testStrikeBeyondBarrier) {
    BOOST_TEST_MESSAGE("Testing options decided by the monitoring date at maturity...");

    // The last monitoring date is maturity: an up call with K >= H or a down put with K <= H
    // pays only where the barrier is crossed at maturity.  Knock-outs are worth zero and
    // knock-ins are worth the vanilla option (Black-Scholes prices computed outside QuantLib).
    CommonVars vars;
    for (Size steps : std::initializer_list<Size>{1, 12, 250}) {
        QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 120.0, Option::Call, steps, 121.0), 1.0e-14);
        QL_CHECK_SMALL(vars.discrete(Barrier::DownOut, 90.0, Option::Put, steps, 89.0), 1.0e-14);
        QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 120.0, Option::Call, steps, 120.0), 1.0e-14);
        QL_CHECK_SMALL(vars.discrete(Barrier::DownOut, 90.0, Option::Put, steps, 90.0), 1.0e-14);
        QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 120.0, Option::Call, steps, 121.0) -
                           4.159597603529679,
                       1.0e-9);
        QL_CHECK_SMALL(vars.discrete(Barrier::DownIn, 90.0, Option::Put, steps, 89.0) -
                           3.912270889154126,
                       1.0e-9);
        QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 120.0, Option::Call, steps, 120.0) -
                           4.374922416029218,
                       1.0e-9);
        QL_CHECK_SMALL(vars.discrete(Barrier::DownIn, 90.0, Option::Put, steps, 90.0) -
                           4.226590998651453,
                       1.0e-9);
    }
}

BOOST_AUTO_TEST_CASE(testStrikeNearBarrier) {
    BOOST_TEST_MESSAGE("Testing options whose strike is close to the barrier...");

    // Strikes on the inside of the exact region, and the other option type with a strike beyond
    // the barrier: all go through the shifted barrier (Reiner-Rubinstein prices at the shifted
    // barrier computed outside QuantLib).  N = 12, T = 1.
    struct Case {
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Real strike;
        Real expected;
    };
    const Case cases[] = {
        {Barrier::UpOut, 120.0, Option::Put, 121.0, 19.2136547821},
        {Barrier::UpOut, 120.0, Option::Put, 130.0, 24.6201889747},
        {Barrier::UpIn, 120.0, Option::Put, 121.0, 2.0248358553},
        {Barrier::UpIn, 120.0, Option::Put, 130.0, 3.6253121572},
        {Barrier::DownIn, 90.0, Option::Call, 89.0, 3.3578422555},
        {Barrier::DownIn, 90.0, Option::Call, 85.0, 4.3743736856},
        {Barrier::DownOut, 90.0, Option::Call, 89.0, 13.9148771838},
        {Barrier::UpOut, 120.0, Option::Call, 119.0, 0.0185922606},
        {Barrier::UpIn, 120.0, Option::Call, 119.0, 4.5813155230},
        {Barrier::DownOut, 90.0, Option::Put, 91.0, 0.0134332196},
        {Barrier::DownIn, 90.0, Option::Put, 91.0, 4.5429404631},
    };

    CommonVars vars;
    for (const auto& c : cases) {
        Real calculated = vars.discrete(c.type, c.barrier, c.optionType, 12, c.strike);
        QL_CHECK_SMALL(calculated - c.expected, 1.0e-8);
    }
}

BOOST_AUTO_TEST_CASE(testKnockOutExceedsContinuous) {
    BOOST_TEST_MESSAGE("Testing the direction of the discrete-monitoring correction...");

    CommonVars vars;
    const Option::Type types[] = {Option::Call, Option::Put};
    for (Option::Type t : types) {
        Real downOutCont = vars.continuous(Barrier::DownOut, 90.0, t);
        Real downOut = vars.discrete(Barrier::DownOut, 90.0, t, 12);
        Real upOutCont = vars.continuous(Barrier::UpOut, 120.0, t);
        Real upOut = vars.discrete(Barrier::UpOut, 120.0, t, 12);
        Real vanilla = vars.continuous(Barrier::DownOut, 1.0e-6, t); // never knocked out
        Real downInCont = vars.continuous(Barrier::DownIn, 90.0, t);
        Real downIn = vars.discrete(Barrier::DownIn, 90.0, t, 12);
        Real upInCont = vars.continuous(Barrier::UpIn, 120.0, t);
        Real upIn = vars.discrete(Barrier::UpIn, 120.0, t, 12);
        BOOST_CHECK(downOut > downOutCont);
        BOOST_CHECK(upOut > upOutCont);
        BOOST_CHECK(downIn < downInCont);
        BOOST_CHECK(upIn < upInCont);
        BOOST_CHECK(downOut < vanilla);
        BOOST_CHECK(upOut < vanilla);
    }
}

BOOST_AUTO_TEST_CASE(testConvergenceToContinuous) {
    BOOST_TEST_MESSAGE("Testing convergence to the continuous barrier price...");

    CommonVars vars;
    const Real continuous = vars.continuous(Barrier::DownOut, 90.0, Option::Call);
    Real previous = QL_MAX_REAL;
    // the shift is proportional to sqrt(dt): each fourfold step halves the gap
    for (Size steps = 4; steps <= 4096; steps *= 4) {
        Real gap = vars.discrete(Barrier::DownOut, 90.0, Option::Call, steps) - continuous;
        BOOST_CHECK(gap > 0.0);
        BOOST_CHECK(gap < previous);
        if (previous < QL_MAX_REAL) {
            Real ratio = gap / previous;
            BOOST_CHECK_MESSAGE(ratio > 0.4 && ratio < 0.65,
                                "gap ratio " << ratio << " at " << steps << " steps");
        }
        previous = gap;
    }
    QL_CHECK_SMALL(previous, 0.15);
}

BOOST_AUTO_TEST_CASE(testAgainstMonteCarlo) {
    BOOST_TEST_MESSAGE("Testing against a Monte Carlo simulation of discrete monitoring...");

    // The seed is fixed, so the run is deterministic.  Over 16 other seeds of this generator
    // these cases stay within 2.6 standard errors of the corrected price and at least 18
    // away from the continuous one.  The correction is an approximation, so the corrected
    // price must lie within four standard errors of the simulation, while the continuous
    // price must lie more than ten standard errors away.
    // The cases are down calls and up puts.  Reverse barriers (up calls, down puts), where
    // the knock-out removes in-the-money payoff, are markedly less accurate at coarse
    // monitoring.  Measured against a 2M+-path exact simulation, S = 100, T = 1: the up-out
    // call with H = 120 and K = 100 is off by about +8-9% at 12 dates and +2.5-3% at 50, the
    // down-out put with H = 90 and K = 100 by about +19% at 12 and +6% at 50.  The relative
    // error grows as the strike approaches the barrier: at 12 dates the up-out call with
    // H = 120 is off by about +27% at K = 110, +80% at K = 115 and 11 times at K = 119, the
    // down-out put with H = 90 by about +55% at K = 95 and 7.5 times at K = 91.
    const Size paths = 200000;
    struct Case {
        Date::serial_type days;
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Size steps;
    };
    const Case cases[] = {
        {365, Barrier::DownOut, 90.0, Option::Call, 12},
        {365, Barrier::DownOut, 90.0, Option::Call, 50},
        {365, Barrier::DownIn, 90.0, Option::Call, 50},
        {365, Barrier::UpIn, 120.0, Option::Put, 12},
        {365, Barrier::UpIn, 120.0, Option::Put, 50},
        {181, Barrier::DownOut, 90.0, Option::Call, 12},
        {181, Barrier::DownIn, 90.0, Option::Call, 12},
        {731, Barrier::UpIn, 120.0, Option::Put, 12},
        {731, Barrier::DownOut, 90.0, Option::Call, 12},
    };

    CommonVars vars;
    for (const auto& c : cases) {
        vars.expiry = vars.today + c.days;
        auto mc = monteCarlo(vars, c.type, c.barrier, c.optionType, c.steps, paths);
        Real corrected = vars.discrete(c.type, c.barrier, c.optionType, c.steps);
        Real continuous = vars.continuous(c.type, c.barrier, c.optionType);
        std::ostringstream where;
        where << " vs simulated " << mc.first << " +- " << mc.second << " at " << c.steps
              << " steps, " << c.days << " days";
        BOOST_CHECK_MESSAGE(std::fabs(corrected - mc.first) < 4.0 * mc.second,
                            "corrected " << corrected << where.str());
        BOOST_CHECK_MESSAGE(std::fabs(continuous - mc.first) > 10.0 * mc.second,
                            "continuous " << continuous << where.str());
    }
}

BOOST_AUTO_TEST_CASE(testInOutParity) {
    BOOST_TEST_MESSAGE("Testing in-out parity of the discrete engine...");

    CommonVars vars;
    const Option::Type types[] = {Option::Call, Option::Put};
    for (Option::Type t : types) {
        VanillaOption vanilla(ext::make_shared<PlainVanillaPayoff>(t, 100.0),
                              ext::make_shared<EuropeanExercise>(vars.expiry));
        vanilla.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(vars.process));
        Real expected = vanilla.NPV();
        for (Size steps : std::initializer_list<Size>{1, 12, 250}) {
            Real down = vars.discrete(Barrier::DownIn, 90.0, t, steps) +
                        vars.discrete(Barrier::DownOut, 90.0, t, steps);
            Real up = vars.discrete(Barrier::UpIn, 120.0, t, steps) +
                      vars.discrete(Barrier::UpOut, 120.0, t, steps);
            QL_CHECK_SMALL(down - expected, 1.0e-10);
            QL_CHECK_SMALL(up - expected, 1.0e-10);
        }
    }
}

BOOST_AUTO_TEST_CASE(testScaling) {
    BOOST_TEST_MESSAGE("Testing that prices scale with the units of spot, strike and barrier...");

    CommonVars vars;
    Real base = vars.discrete(Barrier::UpOut, 120.0, Option::Call, 12, 95.0);
    vars.spot->setValue(1000.0);
    Real scaled = vars.discrete(Barrier::UpOut, 1200.0, Option::Call, 12, 950.0);
    QL_CHECK_SMALL(scaled - 10.0 * base, 1.0e-9);
}

BOOST_AUTO_TEST_CASE(testVolatilityAtStrike) {
    BOOST_TEST_MESSAGE("Testing that the volatility is read at expiry and strike...");

    // a smile: the shift must use the volatility at the option strike
    CommonVars vars;
    std::vector<Date> dates = {vars.today + 6 * Months, vars.expiry, vars.today + 2 * Years};
    std::vector<Real> strikes = {80.0, 100.0, 125.0};
    Matrix vols(3, 3);
    // rows are strikes, columns are dates
    for (Size i = 0; i < 3; ++i) {
        vols[i][0] = 0.30 - 0.04 * Real(i);
        vols[i][1] = 0.32 - 0.04 * Real(i);
        vols[i][2] = 0.34 - 0.04 * Real(i);
    }
    auto surface = ext::make_shared<BlackVarianceSurface>(vars.today, NullCalendar(), dates,
                                                          strikes, vols, vars.dc);
    auto process = ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(vars.spot), vars.process->dividendYield(), vars.process->riskFreeRate(),
        Handle<BlackVolTermStructure>(surface));

    constexpr double beta = 0.5825971579390106;
    for (Real strike : {85.0, 115.0}) {
        Volatility sigma = surface->blackVol(vars.expiry, strike);
        BOOST_REQUIRE(std::fabs(sigma - surface->blackVol(vars.expiry, 100.0)) > 0.01);

        BarrierOption discrete = vars.option(Barrier::DownOut, 90.0, Option::Call, strike);
        discrete.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(process, 12));
        BarrierOption shifted =
            vars.option(Barrier::DownOut, 90.0 * std::exp(-beta * sigma * std::sqrt(1.0 / 12.0)),
                        Option::Call, strike);
        shifted.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(process));
        QL_CHECK_SMALL(discrete.NPV() - shifted.NPV(), 1.0e-12);
    }
}

BOOST_AUTO_TEST_CASE(testErrorMessages) {
    BOOST_TEST_MESSAGE("Testing the input validation of the discrete barrier engine...");

    CommonVars vars;

    BOOST_CHECK_EXCEPTION(AnalyticDiscreteBarrierEngine(vars.process, 0), Error,
                          ExpectedErrorMessage("at least one monitoring date"));

    BarrierOption rebated = vars.option(Barrier::DownOut, 90.0, Option::Call, 100.0, 3.0);
    rebated.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(rebated.NPV(), Error, ExpectedErrorMessage("rebates are not supported"));

    BarrierOption american(Barrier::DownOut, 90.0, 0.0,
                           ext::make_shared<PlainVanillaPayoff>(Option::Call, 100.0),
                           ext::make_shared<AmericanExercise>(vars.today, vars.expiry));
    american.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(american.NPV(), Error,
                          ExpectedErrorMessage("only european style option are supported"));

    BarrierOption digital(Barrier::DownOut, 90.0, 0.0,
                          ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0),
                          ext::make_shared<EuropeanExercise>(vars.expiry));
    digital.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(digital.NPV(), Error, ExpectedErrorMessage("non-plain payoff given"));

    // the same input checks as the continuous engine, also where the price is exact
    BarrierOption negativeStrike = vars.option(Barrier::DownOut, 90.0, Option::Put, -1.0);
    negativeStrike.setPricingEngine(
        ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(negativeStrike.NPV(), Error,
                          ExpectedErrorMessage("strike must be positive"));
    BarrierOption nullStrike = vars.option(Barrier::DownOut, 90.0, Option::Put, 0.0);
    nullStrike.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(nullStrike.NPV(), Error,
                          ExpectedErrorMessage("strike must be positive"));
    vars.spot->setValue(0.0);
    BarrierOption nullSpot = vars.option(Barrier::UpOut, 120.0, Option::Call, 121.0);
    nullSpot.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(nullSpot.NPV(), Error,
                          ExpectedErrorMessage("negative or null underlying given"));
    vars.spot->setValue(100.0);

    BarrierOption americanExact(Barrier::UpOut, 120.0, 0.0,
                                ext::make_shared<PlainVanillaPayoff>(Option::Call, 121.0),
                                ext::make_shared<AmericanExercise>(vars.today, vars.expiry));
    americanExact.setPricingEngine(
        ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(americanExact.NPV(), Error,
                          ExpectedErrorMessage("only european style option are supported"));
    BarrierOption touchedExact = vars.option(Barrier::UpOut, 99.0, Option::Call, 100.0);
    touchedExact.setPricingEngine(
        ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(touchedExact.NPV(), Error, ExpectedErrorMessage("barrier touched"));

    // the barrier given, not the shifted one, decides whether it has been touched
    BarrierOption touched = vars.option(Barrier::DownOut, 101.0, Option::Call);
    touched.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    BOOST_CHECK_EXCEPTION(touched.NPV(), Error, ExpectedErrorMessage("barrier touched"));
}

BOOST_AUTO_TEST_CASE(testQuoteChanges) {
    BOOST_TEST_MESSAGE("Testing that the engine reprices when the market data change...");

    CommonVars vars;
    BarrierOption o = vars.option(Barrier::DownOut, 90.0, Option::Call);
    o.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
    Real before = o.NPV();

    const ext::shared_ptr<SimpleQuote> quotes[] = {vars.spot, vars.rate, vars.dividend, vars.vol};
    const Real bumped[] = {105.0, 0.04, 0.01, 0.30};
    for (Size i = 0; i < 4; ++i) {
        Real old = quotes[i]->value();
        quotes[i]->setValue(bumped[i]);
        Real after = o.NPV();
        Real fresh = vars.discrete(Barrier::DownOut, 90.0, Option::Call, 12);
        BOOST_CHECK(std::fabs(after - before) > 1.0e-3);
        QL_CHECK_SMALL(after - fresh, 1.0e-12);
        quotes[i]->setValue(old);
        QL_CHECK_SMALL(o.NPV() - before, 1.0e-12);
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
