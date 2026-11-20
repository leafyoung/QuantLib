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
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancesurface.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
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
                Handle<Quote>(spot), Handle<YieldTermStructure>(flatRate(today, dividend, dc)),
                Handle<YieldTermStructure>(flatRate(today, rate, dc)),
                Handle<BlackVolTermStructure>(flatVol(today, vol, dc)));
        }

        BarrierOption option(Barrier::Type type,
                             Real barrier,
                             Option::Type optionType,
                             Real strike = 100.0,
                             Real rebate = 0.0) const {
            return BarrierOption(type, barrier, rebate,
                                 ext::make_shared<PlainVanillaPayoff>(optionType, strike),
                                 ext::make_shared<EuropeanExercise>(expiry));
        }

        Real discrete(Barrier::Type type,
                      Real barrier,
                      Option::Type optionType,
                      Size steps,
                      Real strike = 100.0) const {
            BarrierOption o = option(type, barrier, optionType, strike);
            o.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(process, steps));
            return o.NPV();
        }

        Real continuous(Barrier::Type type,
                        Real barrier,
                        Option::Type optionType,
                        Real strike = 100.0) const {
            BarrierOption o = option(type, barrier, optionType, strike);
            o.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(process));
            return o.NPV();
        }

        Real vanilla(Option::Type optionType, Real strike = 100.0) const {
            VanillaOption o(ext::make_shared<PlainVanillaPayoff>(optionType, strike),
                            ext::make_shared<EuropeanExercise>(expiry));
            o.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));
            return o.NPV();
        }
    };

    // names a table row, for the report of a failing check
    std::string describe(Barrier::Type type,
                         Real barrier,
                         Option::Type optionType,
                         Real strike,
                         Size steps,
                         Date::serial_type days,
                         Volatility vol = 0.25) {
        std::ostringstream out;
        out << "type " << type << ", barrier " << barrier << ", option " << optionType
            << ", strike " << strike << ", steps " << steps << ", days " << days << ", vol " << vol;
        return out.str();
    }

    void checkPrice(Real calculated, Real expected, Real tolerance, const std::string& row) {
        BOOST_TEST_CONTEXT(row) {
            QL_CHECK_SMALL(calculated - expected, tolerance);
        }
    }

    // Exact lognormal steps at the N monitoring dates: no time-discretization error.
    // Returns the discounted mean payoff and its standard error.
    std::pair<Real, Real> monteCarlo(const CommonVars& vars,
                                     Barrier::Type type,
                                     Real barrier,
                                     Option::Type optionType,
                                     Real strike,
                                     Size steps,
                                     Size paths) {
        Real s0 = vars.spot->value();
        Real r = vars.rate->value();
        Real q = vars.dividend->value();
        Real sigma = vars.vol->value();
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
            Real payoff = (optionType == Option::Call) ? std::max<Real>(st - strike, 0.0) :
                                                         std::max<Real>(strike - st, 0.0);
            stats.add((hit == knockIn) ? Real(std::exp(-r * maturity) * payoff) : Real(0.0));
        }
        return {stats.mean(), stats.errorEstimate()};
    }

}

BOOST_AUTO_TEST_CASE(testReferenceValues) {
    BOOST_TEST_MESSAGE("Testing discrete barrier prices against the shifted-barrier formula...");

    // Reiner-Rubinstein prices at the shifted barrier H exp(+-beta sigma sqrt(T/N)), computed
    // outside QuantLib.  S = 100, r = 5%, q = 2%, sigma = 25%; maturities of 365, 181 and 731
    // days.  The strikes other than 100 lie close to the barrier, or on the other side of it
    // for the option type that is not decided at maturity.  The boundary between the exact and
    // the shifted-barrier branch is pinned to within 1e-5 of the barrier.
    struct Case {
        Date::serial_type days;
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Real strike;
        Size steps;
        Real expected;
    };
    const Case cases[] = {
        {365, Barrier::DownOut, 90.0, Option::Call, 100.0, 12, 9.5740025184},
        {365, Barrier::DownIn, 90.0, Option::Call, 100.0, 12, 1.5497594097},
        {365, Barrier::DownOut, 90.0, Option::Put, 100.0, 12, 0.2962311188},
        {365, Barrier::DownIn, 90.0, Option::Put, 100.0, 12, 7.9306059287},
        {365, Barrier::UpOut, 120.0, Option::Call, 100.0, 12, 1.3308536713},
        {365, Barrier::UpIn, 120.0, Option::Call, 100.0, 12, 9.7929082568},
        {365, Barrier::UpOut, 120.0, Option::Put, 100.0, 12, 7.9103967300},
        {365, Barrier::UpIn, 120.0, Option::Put, 100.0, 12, 0.3164403175},
        {365, Barrier::DownOut, 90.0, Option::Call, 100.0, 50, 8.9369302463},
        {365, Barrier::UpIn, 120.0, Option::Call, 100.0, 50, 10.1605931817},
        {365, Barrier::DownIn, 90.0, Option::Put, 100.0, 50, 8.0586098363},
        {365, Barrier::UpOut, 120.0, Option::Put, 100.0, 50, 7.7474057970},
        {181, Barrier::DownOut, 90.0, Option::Call, 100.0, 12, 7.1506589134},
        {181, Barrier::DownIn, 90.0, Option::Put, 100.0, 12, 5.6564033695},
        {181, Barrier::UpIn, 120.0, Option::Call, 100.0, 12, 5.4564736195},
        {181, Barrier::UpOut, 120.0, Option::Put, 100.0, 12, 6.1438527113},
        {731, Barrier::DownOut, 90.0, Option::Call, 100.0, 12, 12.5004180490},
        {731, Barrier::DownIn, 90.0, Option::Put, 100.0, 12, 10.3137504186},
        {731, Barrier::UpIn, 120.0, Option::Call, 100.0, 12, 15.3470218519},
        {731, Barrier::UpOut, 120.0, Option::Put, 100.0, 12, 9.3739580301},
        {365, Barrier::UpOut, 120.0, Option::Put, 121.0, 12, 19.2136547821},
        {365, Barrier::UpOut, 120.0, Option::Put, 130.0, 12, 24.6201889747},
        {365, Barrier::UpIn, 120.0, Option::Put, 121.0, 12, 2.0248358553},
        {365, Barrier::UpIn, 120.0, Option::Put, 130.0, 12, 3.6253121572},
        {365, Barrier::DownIn, 90.0, Option::Call, 89.0, 12, 3.3578422555},
        {365, Barrier::DownIn, 90.0, Option::Call, 85.0, 12, 4.3743736856},
        {365, Barrier::DownOut, 90.0, Option::Call, 89.0, 12, 13.9148771838},
        {365, Barrier::UpOut, 120.0, Option::Call, 119.0, 12, 0.0185922606},
        {365, Barrier::UpOut, 120.0, Option::Call, 119.99, 12, 0.0109309881},
        {365, Barrier::UpOut, 120.0, Option::Call, 119.9999, 12, 0.0108676868},
        {365, Barrier::DownOut, 90.0, Option::Put, 90.005, 12, 0.0066575506},
        {365, Barrier::DownOut, 90.0, Option::Put, 90.0001, 12, 0.0066315532},
        {365, Barrier::DownIn, 90.0, Option::Put, 90.0001, 12, 4.2199916487},
        {365, Barrier::DownOut, 90.0, Option::Put, 90.00001, 12, 0.0066310763},
        {365, Barrier::DownIn, 90.0, Option::Put, 90.00001, 12, 4.2199631426},
        {365, Barrier::DownOut, 90.0, Option::Put, 90.5, 12, 0.0096422182},
        {365, Barrier::UpIn, 120.0, Option::Call, 119.0, 12, 4.5813155230},
        {365, Barrier::DownOut, 90.0, Option::Put, 91.0, 12, 0.0134332196},
        {365, Barrier::DownIn, 90.0, Option::Put, 91.0, 12, 4.5429404631},
    };

    CommonVars vars;
    for (const auto& c : cases) {
        vars.expiry = vars.today + c.days;
        Real calculated = vars.discrete(c.type, c.barrier, c.optionType, c.steps, c.strike);
        checkPrice(calculated, c.expected, 1.0e-8,
                   describe(c.type, c.barrier, c.optionType, c.strike, c.steps, c.days));
    }
}

BOOST_AUTO_TEST_CASE(testStrikeBeyondBarrier) {
    BOOST_TEST_MESSAGE("Testing options decided by the monitoring date at maturity...");

    // The last monitoring date is maturity: an up call with K >= H or a down put with K <= H
    // pays only where the barrier is crossed at maturity.  Knock-outs are worth zero and
    // knock-ins are worth the vanilla option (Black-Scholes prices computed outside QuantLib).
    struct Case {
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Real strike;
        Real expected;
        Real tolerance;
    };
    const Case cases[] = {
        {Barrier::UpOut, 120.0, Option::Call, 121.0, 0.0, 1.0e-14},
        {Barrier::DownOut, 90.0, Option::Put, 89.0, 0.0, 1.0e-14},
        {Barrier::UpOut, 120.0, Option::Call, 120.0, 0.0, 1.0e-14},
        {Barrier::DownOut, 90.0, Option::Put, 90.0, 0.0, 1.0e-14},
        {Barrier::UpIn, 120.0, Option::Call, 121.0, 4.159597603529679, 1.0e-9},
        {Barrier::DownIn, 90.0, Option::Put, 89.0, 3.912270889154126, 1.0e-9},
        {Barrier::UpIn, 120.0, Option::Call, 120.0, 4.374922416029218, 1.0e-9},
        {Barrier::DownIn, 90.0, Option::Put, 90.0, 4.226590998651453, 1.0e-9},
    };

    CommonVars vars;
    for (Size steps : std::vector<Size>{1, 12, 250}) {
        for (const auto& c : cases) {
            Real calculated = vars.discrete(c.type, c.barrier, c.optionType, steps, c.strike);
            checkPrice(calculated, c.expected, c.tolerance,
                       describe(c.type, c.barrier, c.optionType, c.strike, steps, 365));
        }
    }
}

BOOST_AUTO_TEST_CASE(testCorrectionDirection) {
    BOOST_TEST_MESSAGE("Testing the direction of the discrete-monitoring correction...");

    CommonVars vars;
    const Option::Type types[] = {Option::Call, Option::Put};
    for (Option::Type t : types) {
        Real downOutCont = vars.continuous(Barrier::DownOut, 90.0, t);
        Real downOut = vars.discrete(Barrier::DownOut, 90.0, t, 12);
        Real upOutCont = vars.continuous(Barrier::UpOut, 120.0, t);
        Real upOut = vars.discrete(Barrier::UpOut, 120.0, t, 12);
        Real vanilla = vars.vanilla(t);
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

    // The corrected price must lie within four standard errors of the simulation (fixed seed),
    // while the continuous price must lie more than ten standard errors away.  The cases are
    // down calls and up puts: reverse barriers (up calls, down puts) are markedly less accurate
    // at coarse monitoring, increasingly so as the strike approaches the barrier, and regular
    // barriers are approximate too at one or two monitoring dates.
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
        auto mc = monteCarlo(vars, c.type, c.barrier, c.optionType, 100.0, c.steps, paths);
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
        Real expected = vars.vanilla(t);
        for (Size steps : std::vector<Size>{1, 12, 250}) {
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

    // expiries on the one-year node and between the nodes, where the surface interpolates;
    // beta is restated here on purpose (independent of the engine's constant)
    constexpr double beta = 0.5825971579390106;
    const Date nodeExpiry = vars.expiry;
    for (Date expiry : {nodeExpiry, vars.today + 9 * Months, vars.today + 18 * Months}) {
        vars.expiry = expiry;
        Real maturity = vars.dc.yearFraction(vars.today, expiry);
        for (Real strike : std::vector<Real>{85.0, 115.0}) {
            Volatility sigma = surface->blackVol(expiry, strike);
            BOOST_REQUIRE(std::fabs(sigma - surface->blackVol(expiry, 100.0)) > 0.01);

            BarrierOption discrete = vars.option(Barrier::DownOut, 90.0, Option::Call, strike);
            discrete.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(process, 12));
            BarrierOption shifted = vars.option(
                Barrier::DownOut, 90.0 * std::exp(-beta * sigma * std::sqrt(maturity / 12.0)),
                Option::Call, strike);
            shifted.setPricingEngine(ext::make_shared<AnalyticBarrierEngine>(process));
            QL_CHECK_SMALL(discrete.NPV() - shifted.NPV(), 1.0e-12);
        }
    }
}

BOOST_AUTO_TEST_CASE(testZeroVolatility) {
    BOOST_TEST_MESSAGE("Testing the zero-volatility limit...");

    // The underlying follows its forward, 100 exp(0.03 t): it reaches 102 before maturity and
    // stays above 90 (values computed outside QuantLib).
    CommonVars vars;
    vars.vol->setValue(0.0);
    const Real vanilla = 2.8969248806;
    QL_CHECK_SMALL(vars.discrete(Barrier::DownOut, 90.0, Option::Call, 12) - vanilla, 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 102.0, Option::Call, 12), 1.0e-9);
    // the forward at maturity is 103.0455: still above 103
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 103.0, Option::Call, 12), 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 103.0, Option::Call, 12) - vanilla, 1.0e-9);
    // a barrier 0.0053% below the forward 103.0455: it is still hit at maturity
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 103.04, Option::Call, 12), 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 103.04, Option::Call, 12) - vanilla, 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 102.0, Option::Call, 12) - vanilla, 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 102.0, Option::Call, 12, 102.5), 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 102.0, Option::Call, 12, 102.5) - 0.5188513194,
                   1.0e-9);
}

BOOST_AUTO_TEST_CASE(testLargeVolatility) {
    BOOST_TEST_MESSAGE("Testing a large volatility, where the barrier shift is about 31%...");

    // sigma = 160%, T = 1, N = 12: shift = exp(beta sigma sqrt(T/N)) = 1.3088.  Reiner-Rubinstein
    // prices at the shifted barrier computed outside QuantLib.
    CommonVars vars;
    vars.vol->setValue(1.6);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 103.0, Option::Call, 12) - 0.0177526624, 1.0e-8);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 120.0, Option::Call, 12) - 0.0902445760, 1.0e-8);
    QL_CHECK_SMALL(vars.discrete(Barrier::DownOut, 90.0, Option::Call, 12) - 29.0453923749, 1.0e-8);
    QL_CHECK_SMALL(vars.discrete(Barrier::DownOut, 90.0, Option::Put, 12) - 0.0431694841, 1.0e-8);
}

BOOST_AUTO_TEST_CASE(testVolatilityClock) {
    BOOST_TEST_MESSAGE("Testing that the time to expiry is measured on the volatility clock...");

    // Actual/360 volatility surface over Actual/365 (Fixed) rate curves: the monitoring step
    // is T_vol/N with T_vol = 365/360, while discounting runs over T = 1.  Reference prices at
    // the barrier moved by exp(+-beta sigma sqrt(T_vol/N)), computed outside QuantLib.  This
    // pins the convention of AnalyticBarrierEngine, which the corrected price inherits: the
    // drift uses the volatility as quoted, the variance is sigma^2 T_vol.
    CommonVars vars;
    auto process = ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(vars.spot), vars.process->dividendYield(), vars.process->riskFreeRate(),
        Handle<BlackVolTermStructure>(flatVol(vars.today, vars.vol, Actual360())));

    struct Case {
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Real expected;
    };
    const Case cases[] = {
        {Barrier::DownOut, 90.0, Option::Call, 9.6143073983},
        {Barrier::DownIn, 90.0, Option::Put, 7.9970499789},
        {Barrier::UpIn, 120.0, Option::Call, 9.8793277723},
        {Barrier::UpOut, 120.0, Option::Put, 7.9646482768},
    };
    for (const auto& c : cases) {
        BarrierOption o = vars.option(c.type, c.barrier, c.optionType);
        o.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(process, 12));
        QL_CHECK_SMALL(o.NPV() - c.expected, 1.0e-8);
    }
}

BOOST_AUTO_TEST_CASE(testCoarseAndShortMonitoring) {
    BOOST_TEST_MESSAGE("Testing coarse and fine monitoring and extreme inputs...");

    // Reiner-Rubinstein prices at the shifted barrier computed outside QuantLib.
    struct Case {
        Date::serial_type days;
        Volatility vol;
        Size steps;
        Barrier::Type type;
        Real barrier;
        Option::Type optionType;
        Real expected;
    };
    const Case cases[] = {
        // two monitoring dates
        {365, 0.25, 2, Barrier::DownOut, 90.0, Option::Call, 10.6065575706},
        {365, 0.25, 2, Barrier::UpOut, 120.0, Option::Put, 8.1416688799},
        {365, 0.25, 2, Barrier::DownIn, 90.0, Option::Put, 7.2792403915},
        // a single monitoring date at maturity, T = 731/365
        {731, 0.25, 1, Barrier::DownOut, 90.0, Option::Call, 15.5100009519},
        {731, 0.25, 1, Barrier::UpOut, 120.0, Option::Put, 10.3391477790},
        {731, 0.25, 1, Barrier::UpIn, 120.0, Option::Call, 12.7216714278},
        // seven days: monitoring steps of 0.0016 years (N = 12), 0.00008 (N = 252) and
        // 1e-6 (N = 20000)
        {7, 0.25, 12, Barrier::DownOut, 98.0, Option::Call, 1.3072368442},
        {7, 0.25, 252, Barrier::DownOut, 98.0, Option::Call, 1.2281225948},
        {7, 0.25, 20000, Barrier::DownOut, 98.0, Option::Call, 1.2021028353},
        // thirty days
        {30, 0.25, 12, Barrier::DownOut, 95.0, Option::Call, 2.8618269588},
        {30, 0.25, 12, Barrier::UpOut, 105.0, Option::Put, 2.6012851255},
        {30, 0.25, 12, Barrier::UpIn, 105.0, Option::Call, 2.6530467512},
        {30, 0.25, 12, Barrier::DownIn, 95.0, Option::Put, 2.3747297284},
        // volatility of 350% with four dates: the barrier moves by a factor of 2.77
        {365, 3.5, 4, Barrier::DownOut, 90.0, Option::Call, 65.3126934883},
        {365, 3.5, 4, Barrier::UpOut, 120.0, Option::Call, 0.1277573242},
        {365, 3.5, 4, Barrier::UpOut, 120.0, Option::Put, 65.1878997960},
        {365, 3.5, 4, Barrier::DownIn, 90.0, Option::Call, 24.9711068684},
    };

    CommonVars vars;
    for (const auto& c : cases) {
        vars.expiry = vars.today + c.days;
        vars.vol->setValue(c.vol);
        checkPrice(vars.discrete(c.type, c.barrier, c.optionType, c.steps), c.expected, 1.0e-8,
                   describe(c.type, c.barrier, c.optionType, 100.0, c.steps, c.days, c.vol));
    }
}

BOOST_AUTO_TEST_CASE(testVolatilityReferenceDate) {
    BOOST_TEST_MESSAGE("Testing a volatility surface with a later reference date...");

    // The surface starts 180 days after the evaluation date, so the monitoring step is
    // T_vol/N with T_vol = 185/365 while discounting runs over T = 1 (Reiner-Rubinstein prices
    // with the barrier moved by exp(+-beta sigma sqrt(T_vol/N)), computed outside QuantLib).
    CommonVars vars;
    auto surface = ext::make_shared<BlackConstantVol>(vars.today + 180, NullCalendar(),
                                                      Handle<Quote>(vars.vol), vars.dc);
    auto process = ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(vars.spot), vars.process->dividendYield(), vars.process->riskFreeRate(),
        Handle<BlackVolTermStructure>(surface));

    BarrierOption downOut = vars.option(Barrier::DownOut, 90.0, Option::Call);
    downOut.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(process, 12));
    QL_CHECK_SMALL(downOut.NPV() - 7.7642968072, 1.0e-8);
    BarrierOption upIn = vars.option(Barrier::UpIn, 120.0, Option::Put);
    upIn.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(process, 12));
    QL_CHECK_SMALL(upIn.NPV() - 0.0343192764, 1.0e-8);
}

BOOST_AUTO_TEST_CASE(testSpotOnBarrier) {
    BOOST_TEST_MESSAGE("Testing a spot on the barrier...");

    CommonVars vars;

    // zero volatility: the barrier is hit at the valuation date, as in the continuous engine
    vars.vol->setValue(0.0);
    const Real vanilla = 2.8969248806;
    QL_CHECK_SMALL(vars.discrete(Barrier::DownOut, 100.0, Option::Call, 12), 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::DownIn, 100.0, Option::Call, 12) - vanilla, 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 100.0, Option::Call, 12), 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 100.0, Option::Call, 12) - vanilla, 1.0e-9);
    // an up call with K < H goes through the shifted barrier, not the exact branch
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 100.0, Option::Call, 12, 95.0), 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 100.0, Option::Call, 12, 95.0) - 7.6530720031,
                   1.0e-9);

    // a negative carry takes the forward below the up barrier, which the spot is on: still hit
    vars.rate->setValue(0.01);
    vars.dividend->setValue(0.03);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 100.0, Option::Call, 12, 95.0), 1.0e-9);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpIn, 100.0, Option::Call, 12, 95.0) - 2.9898191487,
                   1.0e-9);
    vars.rate->setValue(0.05);
    vars.dividend->setValue(0.02);

    // positive volatility: the shifted barrier is away from the spot, so the option prices
    // (Reiner-Rubinstein prices at the shifted barrier computed outside QuantLib)
    vars.vol->setValue(0.25);
    QL_CHECK_SMALL(vars.discrete(Barrier::DownOut, 100.0, Option::Call, 12) - 4.1893931494, 1.0e-8);
    QL_CHECK_SMALL(vars.discrete(Barrier::UpOut, 100.0, Option::Put, 12) - 2.9290338181, 1.0e-8);
}

BOOST_AUTO_TEST_CASE(testErrorMessages) {
    BOOST_TEST_MESSAGE("Testing the input validation of the discrete barrier engine...");

    CommonVars vars;

    BOOST_CHECK_EXCEPTION(AnalyticDiscreteBarrierEngine(vars.process, 0), Error,
                          ExpectedErrorMessage("at least one monitoring date is required"));

    auto european = [&]() { return ext::make_shared<EuropeanExercise>(vars.expiry); };
    auto american = [&]() { return ext::make_shared<AmericanExercise>(vars.today, vars.expiry); };
    auto bermudan = [&]() {
        return ext::make_shared<BermudanExercise>(
            std::vector<Date>{vars.today + 6 * Months, vars.expiry});
    };
    auto plain = [](Option::Type type, Real strike) {
        return ext::make_shared<PlainVanillaPayoff>(type, strike);
    };
    // pricing the option with the discrete engine must fail with the given message
    auto check = [&](Barrier::Type type, Real barrier, Real rebate,
                     const ext::shared_ptr<StrikedTypePayoff>& payoff,
                     const ext::shared_ptr<Exercise>& exercise, const std::string& message) {
        BarrierOption o(type, barrier, rebate, payoff, exercise);
        o.setPricingEngine(ext::make_shared<AnalyticDiscreteBarrierEngine>(vars.process, 12));
        BOOST_CHECK_EXCEPTION(o.NPV(), Error, ExpectedErrorMessage(message));
    };
    const auto digital = ext::make_shared<CashOrNothingPayoff>(Option::Call, 100.0, 1.0);
    const std::string rebateMessage =
        "rebates are not supported by the discrete-monitoring correction";
    const std::string exerciseMessage = "only european style option are supported";
    const std::string strikeMessage = "strike must be positive";
    const std::string spotMessage = "negative or null underlying given";
    const std::string touchedMessage = "barrier touched";

    check(Barrier::DownOut, 90.0, 3.0, plain(Option::Call, 100.0), european(), rebateMessage);
    check(Barrier::DownOut, 90.0, 0.01, plain(Option::Call, 100.0), european(), rebateMessage);
    check(Barrier::DownOut, 90.0, 1.0e-9, plain(Option::Call, 100.0), european(), rebateMessage);
    check(Barrier::DownOut, 90.0, 1.0e-13, plain(Option::Call, 100.0), european(), rebateMessage);
    check(Barrier::DownOut, 90.0, 0.0, plain(Option::Call, 100.0), american(), exerciseMessage);
    check(Barrier::DownOut, 90.0, 0.0, digital, european(), "non-plain payoff given");

    // the same input checks as the continuous engine, also where the price is exact
    check(Barrier::DownOut, 90.0, 0.0, plain(Option::Put, -1.0), european(), strikeMessage);
    check(Barrier::DownOut, 90.0, 0.0, plain(Option::Put, 0.0), european(), strikeMessage);
    check(Barrier::UpOut, 120.0, 0.0, plain(Option::Call, 121.0), american(), exerciseMessage);
    check(Barrier::UpOut, 120.0, 0.0, plain(Option::Call, 121.0), bermudan(), exerciseMessage);
    check(Barrier::UpOut, 99.0, 0.0, plain(Option::Call, 100.0), european(), touchedMessage);
    // the barrier given, not the shifted one, decides whether it has been touched
    check(Barrier::DownOut, 101.0, 0.0, plain(Option::Call, 100.0), european(), touchedMessage);

    // the order of the checks: rebate, payoff, strike, exercise, spot, barrier touched
    check(Barrier::DownOut, 90.0, 3.0, plain(Option::Call, 100.0), american(), rebateMessage);
    check(Barrier::DownOut, 90.0, 3.0, digital, european(), rebateMessage);
    check(Barrier::DownOut, 90.0, 0.0, plain(Option::Call, -1.0), american(), strikeMessage);

    vars.spot->setValue(0.0);
    check(Barrier::UpOut, 120.0, 0.0, plain(Option::Call, 121.0), european(), spotMessage);
    // strike is checked before the spot, exercise before the spot, the spot before the barrier
    check(Barrier::DownOut, 90.0, 0.0, plain(Option::Put, 0.0), european(), strikeMessage);
    check(Barrier::DownOut, 90.0, 0.0, plain(Option::Call, 100.0), american(), exerciseMessage);
    check(Barrier::DownOut, 90.0, 0.0, plain(Option::Call, 100.0), european(), spotMessage);
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
