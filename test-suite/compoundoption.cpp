/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2009 Dimitri Reiswich

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
#include <ql/instruments/compoundoption.hpp>
#include <ql/pricingengines/exotic/analyticcompoundoptionengine.hpp>
#include <ql/instruments/europeanoption.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
#include <ql/shared_ptr.hpp>
#include <ql/time/calendars/target.hpp>
#include <ql/time/calendars/nullcalendar.hpp>
#include <ql/time/daycounters/actual360.hpp>
#include <ql/quotes/simplequote.hpp>
#include <ql/termstructures/yield/flatforward.hpp>
#include <ql/termstructures/yield/zerocurve.hpp>
#include <ql/termstructures/volatility/equityfx/blackvariancecurve.hpp>
#include <ql/time/daycounters/actual365fixed.hpp>
#include <ql/termstructures/volatility/equityfx/blackconstantvol.hpp>
#include <ql/utilities/dataformatters.hpp>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(CompoundOptionTests)

#undef REPORT_FAILURE
#define REPORT_FAILURE(greekName, payoffM, payoffD, exerciseM,    \
                       exerciseD, s, q, r, today,                 \
                       v, expected, calculated, error, tolerance) \
            BOOST_FAIL(\
               "\nmother option type:   " << payoffM->optionType() << \
               "\ndaughter option type: " << payoffD->optionType() << \
               "\nspot value:           " << s << \
               "\nstrike mother:        " << payoffM->strike() << \
               "\nstrike daughter:      " << payoffD->strike() << \
               "\ndividend yield:       " << io::rate(q) << \
               "\nrisk-free rate:       " << io::rate(r) << \
               "\nreference date:       " << today << \
               "\nmaturity mother:      " << exerciseM->lastDate() << \
               "\nmaturity daughter:    " << exerciseD->lastDate() << \
               "\nvolatility:           " << io::volatility(v) << \
               "\n  expected " << greekName << ": " << expected << \
               "\ncalculated " << greekName << ": " << calculated << \
               "\nerror:                " << error << \
               "\ntolerance:            " << tolerance);

struct CompoundOptionData {
    Option::Type typeMother;
    Option::Type typeDaughter;
    Real strikeMother;
    Real strikeDaughter;
    Real s;        // spot
    Rate q;        // dividend
    Rate r;        // risk-free rate
    Time tMother;  // time to maturity
    Time tDaughter;// time to maturity
    Volatility v;  // volatility
    Real npv;   // expected result
    Real tol;      // tolerance
    Real delta;
    Real gamma;
    Real vega;
    Real theta;
};


BOOST_AUTO_TEST_CASE(testPutCallParity){

    BOOST_TEST_MESSAGE("Testing compound-option put-call parity...");

    // Test Put Call Parity for compound options.
    // Formula taken from: "Foreign Exchange Risk", Wystup, Risk 2002
    // Page 81, Equation 9.5


    CompoundOptionData values[] = {
        // type Mother, typeDaughter, strike Mother, strike Daughter,  spot,    q,    r,    t Mother, t Daughter,  vol
        { Option::Put, Option::Call,  50.0,            520.0   ,      500.0,   0.03, 0.08,  0.25,     0.5,        0.35},
        { Option::Call, Option::Call,  50.0,           520.0   ,      500.0,   0.03, 0.08,  0.25,     0.5,        0.35},
        { Option::Call, Option::Put,  50.0,            520.0   ,      500.0,   0.03, 0.08,  0.25,     0.5,        0.35},
        { Option::Call, Option::Call,  0.05,           1.14   ,      1.20,  0.0, 0.01,  0.5,     2.0,         0.11},
        { Option::Call, Option::Put ,  0.05,           1.14   ,      1.20,  0.0, 0.01,  0.5,     2.0,         0.11},
        { Option::Call, Option::Call,  10.0,           122.0   ,      120.0,    0.06, 0.02,  0.1,     0.7,        0.22},
        { Option::Call, Option::Put,  10.0,           122.0   ,      120.0,     0.06, 0.02,  0.1,     0.7,        0.22},
        { Option::Call, Option::Call,  0.4,           8.2   ,      8.0,     0.05, 0.00,  2.0,     3.0,        0.08},
        { Option::Call, Option::Put,  0.4,           8.2   ,      8.0,  0.05, 0.00,  2.0,     3.0,        0.08},
        { Option::Call, Option::Call,  0.02,           1.6   ,      1.6,    0.013, 0.022,  0.45,     0.5,        0.17},
        { Option::Call, Option::Put,  0.02,           1.6   ,      1.6,     0.013, 0.022,  0.45,     0.5,         0.17},
    };

    Calendar calendar = TARGET();

    DayCounter dc = Actual360();
    Date todaysDate = Settings::instance().evaluationDate();

    ext::shared_ptr<SimpleQuote> spot = ext::make_shared<SimpleQuote>(0.0);
    ext::shared_ptr<SimpleQuote> rRate = ext::make_shared<SimpleQuote>(0.0);
    ext::shared_ptr<SimpleQuote> qRate = ext::make_shared<SimpleQuote>(0.0);
    ext::shared_ptr<SimpleQuote> vol = ext::make_shared<SimpleQuote>(0.0);

    ext::shared_ptr<YieldTermStructure> rTS(
              new FlatForward(0, NullCalendar(), Handle<Quote>(rRate), dc));

    ext::shared_ptr<YieldTermStructure> qTS(
              new FlatForward(0, NullCalendar(), Handle<Quote>(qRate), dc));

    ext::shared_ptr<BlackVolTermStructure> volTS(
                              new BlackConstantVol(todaysDate, NullCalendar(),
                                                   Handle<Quote>(vol), dc));

    for (auto& value : values) {

        ext::shared_ptr<StrikedTypePayoff> payoffMotherCall(
            new PlainVanillaPayoff(Option::Call, value.strikeMother));

        ext::shared_ptr<StrikedTypePayoff> payoffMotherPut(
            new PlainVanillaPayoff(Option::Put, value.strikeMother));

        ext::shared_ptr<StrikedTypePayoff> payoffDaughter(
            new PlainVanillaPayoff(value.typeDaughter, value.strikeDaughter));

        Date matDateMom = todaysDate + timeToDays(value.tMother);
        Date matDateDaughter = todaysDate + timeToDays(value.tDaughter);

        ext::shared_ptr<Exercise> exerciseCompound(
                                            new EuropeanExercise(matDateMom));
        ext::shared_ptr<Exercise> exerciseDaughter(
                                       new EuropeanExercise(matDateDaughter));

        spot->setValue(value.s);
        qRate->setValue(value.q);
        rRate->setValue(value.r);
        vol->setValue(value.v);

        CompoundOption compoundOptionCall(payoffMotherCall,exerciseCompound,
                                          payoffDaughter, exerciseDaughter);

        CompoundOption compoundOptionPut(payoffMotherPut,exerciseCompound,
                                         payoffDaughter, exerciseDaughter);

        VanillaOption vanillaOption(EuropeanOption(payoffDaughter,
                                                   exerciseDaughter));

        ext::shared_ptr<BlackScholesMertonProcess> stochProcess(
            new BlackScholesMertonProcess(
                      Handle<Quote>(spot),
                      Handle<YieldTermStructure>(qTS),
                      Handle<YieldTermStructure>(rTS),
                      Handle<BlackVolTermStructure>(volTS)));


        ext::shared_ptr<PricingEngine> engineCompound(
                              new AnalyticCompoundOptionEngine(stochProcess));

        ext::shared_ptr<PricingEngine> engineEuropean(
                                     new AnalyticEuropeanEngine(stochProcess));

        compoundOptionCall.setPricingEngine(engineCompound);
        compoundOptionPut.setPricingEngine(engineCompound);
        vanillaOption.setPricingEngine(engineEuropean);

        Real discFact=rTS->discount(matDateMom);
        Real discStrike = value.strikeMother * discFact;

        Real calculated =
            compoundOptionCall.NPV() + discStrike - compoundOptionPut.NPV()
            - vanillaOption.NPV();

        Real expected=0.0;
        Real error=std::abs(calculated-expected);
        Real tolerance=1.0e-8;

        if(error>tolerance){
            REPORT_FAILURE("put call parity", payoffMotherCall, payoffDaughter, exerciseCompound,
                           exerciseDaughter, value.s, value.q, value.r, todaysDate, value.v,
                           value.delta, calculated, error, tolerance);
        }
    }
}

BOOST_AUTO_TEST_CASE(testValues){

    BOOST_TEST_MESSAGE("Testing compound-option values and greeks...");

    CompoundOptionData values[] = {
        // type Mother, typeDaughter, strike Mother, strike Daughter,  spot,    q,    r,    t Mother, t Daughter,  vol,   value,    tol, delta, gamma, vega, theta
        // Tolerance is taken to be pretty high with 1.0e-3, since the price/theta is very sensitive with respect to
        // the implementation of the bivariate normal - which differs in the various implementations.
        // Option Value Taken from Haug 2007, Greeks from www.sitmo.com
        { Option::Put, Option::Call,  50.0,            520.0   ,      500.0,   0.03, 0.08,  0.25,     0.5,        0.35,  21.1965,   1.0e-3, -0.1966,0.0007, -32.1241, -3.3837},
        //*********************************************************
        // Option Values and Greeks taken from www.sitmo.com
        { Option::Call, Option::Call,  50.0,           520.0   ,      500.0,   0.03, 0.08,  0.25,     0.5,        0.35,  17.5945,   1.0e-3,  0.3219,0.0038, 106.5185, -65.1614},
        { Option::Call, Option::Put,  50.0,            520.0   ,      500.0,   0.03, 0.08,  0.25,     0.5,        0.35,  18.7128,   1.0e-3,  -0.2906,0.0036, 103.3856, -46.6982},
        { Option::Put, Option::Put,  50.0,            520.0   ,      500.0,    0.03, 0.08,  0.25,     0.5,        0.35,  15.2601,   1.0e-3,  0.1760,0.0005, -35.2570, -10.1126},
        // type Mother, typeDaughter, strike Mother, strike Daughter,  spot,    q,    r,    t Mother, t Daughter,  vol,   value,    tol, delta, gamma, vega, theta
        { Option::Call, Option::Call,  0.05,           1.14   ,      1.20,  0.0, 0.01,  0.5,     2.0,         0.11,  0.0729,   1.0e-3,  0.6614,2.5762, 0.5812, -0.0297},
        { Option::Call, Option::Put ,  0.05,           1.14   ,      1.20,  0.0, 0.01,  0.5,     2.0,         0.11,  0.0074,   1.0e-3,  -0.1334,1.9681, 0.2933, -0.0155},
        { Option::Put  ,Option::Call,  0.05,           1.14   ,      1.20,  0.0, 0.01,  0.5,     2.0,         0.11,  0.0021,   1.0e-3,  -0.0426,0.7252, -0.0052, -0.0058},
        { Option::Put, Option::Put ,  0.05,           1.14   ,      1.20,   0.0, 0.01,  0.5,     2.0,         0.11,  0.0192,   1.0e-3,  0.1626,0.1171, -0.2931, -0.0028},
        // type Mother, typeDaughter, strike Mother, strike Daughter,  spot,    q,    r,    t Mother, t Daughter,  vol,   value,    tol, delta, gamma, vega, theta
        { Option::Call, Option::Call,  10.0,           122.0   ,      120.0,    0.06, 0.02,  0.1,     0.7,        0.22,  0.4419,   1.0e-3,  0.1049,0.0195, 11.3368, -6.2871},
        { Option::Call, Option::Put,  10.0,           122.0   ,      120.0,     0.06, 0.02,  0.1,     0.7,        0.22,  2.6112,   1.0e-3,  -0.3618,0.0337, 28.4843, -13.4124},
        { Option::Put, Option::Call,  10.0,           122.0   ,      120.0,     0.06, 0.02,  0.1,     0.7,        0.22,  4.1616,   1.0e-3,  -0.3174,0.0024, -26.6403, -2.2720},
        { Option::Put, Option::Put,  10.0,           122.0   ,      120.0,  0.06, 0.02,  0.1,     0.7,        0.22,  1.0914,   1.0e-3,  0.1748,0.0165, -9.4928, -4.8995},
        //*********************************************************
        //*********************************************************
        // Option Values and Greeks taken from mathfinance VBA implementation
        // type Mother, typeDaughter, strike Mother, strike Daughter,  spot,    q,    r,    t Mother, t Daughter,  vol,   value,    tol, delta, gamma, vega, theta
        { Option::Call, Option::Call,  0.4,           8.2   ,      8.0,  0.05, 0.00,  2.0,     3.0,        0.08,  0.0099,   1.0e-3,  0.0285,0.0688, 0.7764, -0.0027},
        { Option::Call, Option::Put,  0.4,           8.2   ,      8.0,   0.05, 0.00,  2.0,     3.0,        0.08,  0.9826,   1.0e-3,  -0.7224,0.2158, 2.7279, -0.3332},
        { Option::Put, Option::Call,  0.4,           8.2   ,      8.0,   0.05, 0.00,  2.0,     3.0,        0.08,  0.3585,   1.0e-3,  -0.0720,-0.0835, -1.5633, -0.0117},
        { Option::Put, Option::Put,  0.4,           8.2   ,      8.0,    0.05, 0.00,  2.0,     3.0,        0.08,  0.0168,   1.0e-3,  0.0378, 0.0635, 0.3882, 0.0021},
        // type Mother, typeDaughter, strike Mother, strike Daughter,  spot,    q,    r,    t Mother, t Daughter,  vol,   value,    tol, delta, gamma, vega, theta
        { Option::Call, Option::Call,  0.02,           1.6   ,      1.6,     0.013, 0.022,  0.45,     0.5,         0.17,  0.0680,   1.0e-3,  0.4937,2.1271, 0.4418, -0.0843},
        { Option::Call, Option::Put,  0.02,           1.6   ,      1.6,  0.013, 0.022,  0.45,     0.5,         0.17,  0.0605,   1.0e-3,  -0.4169,2.0836, 0.4330, -0.0697},
        { Option::Put, Option::Call,  0.02,           1.6   ,      1.6,  0.013, 0.022,  0.45,     0.5,         0.17,  0.0081,   1.0e-3,  -0.0417,0.0761, -0.0045, -0.0020},
        { Option::Put, Option::Put,  0.02,           1.6   ,      1.6,   0.013, 0.022,  0.45,     0.5,         0.17,  0.0078,   1.0e-3,   0.0413,0.0326, -0.0133, -0.0016}
    };

    Calendar calendar = TARGET();

    DayCounter dc = Actual360();
    Date todaysDate = Settings::instance().evaluationDate();

    ext::shared_ptr<SimpleQuote> spot = ext::make_shared<SimpleQuote>(0.0);
    ext::shared_ptr<SimpleQuote> rRate = ext::make_shared<SimpleQuote>(0.0);
    ext::shared_ptr<SimpleQuote> qRate = ext::make_shared<SimpleQuote>(0.0);
    ext::shared_ptr<SimpleQuote> vol = ext::make_shared<SimpleQuote>(0.0);

    ext::shared_ptr<YieldTermStructure> rTS(
              new FlatForward(0, NullCalendar(), Handle<Quote>(rRate), dc));

    ext::shared_ptr<YieldTermStructure> qTS(
              new FlatForward(0, NullCalendar(), Handle<Quote>(qRate), dc));

    ext::shared_ptr<BlackVolTermStructure> volTS(
                              new BlackConstantVol(todaysDate, NullCalendar(),
                                                   Handle<Quote>(vol), dc));

    for (auto& value : values) {

        ext::shared_ptr<StrikedTypePayoff> payoffMother(
            new PlainVanillaPayoff(value.typeMother, value.strikeMother));

        ext::shared_ptr<StrikedTypePayoff> payoffDaughter(
            new PlainVanillaPayoff(value.typeDaughter, value.strikeDaughter));

        Date matDateMom = todaysDate + timeToDays(value.tMother);
        Date matDateDaughter = todaysDate + timeToDays(value.tDaughter);

        ext::shared_ptr<Exercise> exerciseMother(
                                            new EuropeanExercise(matDateMom));
        ext::shared_ptr<Exercise> exerciseDaughter(
                                       new EuropeanExercise(matDateDaughter));

        spot->setValue(value.s);
        qRate->setValue(value.q);
        rRate->setValue(value.r);
        vol->setValue(value.v);

        CompoundOption compoundOption(payoffMother,exerciseMother,
                                      payoffDaughter, exerciseDaughter);

        ext::shared_ptr<BlackScholesMertonProcess> stochProcess(
            new BlackScholesMertonProcess(
                      Handle<Quote>(spot),
                      Handle<YieldTermStructure>(qTS),
                      Handle<YieldTermStructure>(rTS),
                      Handle<BlackVolTermStructure>(volTS)));

        ext::shared_ptr<PricingEngine> engineCompound(
                              new AnalyticCompoundOptionEngine(stochProcess));

        compoundOption.setPricingEngine(engineCompound);

        Real calculated = compoundOption.NPV();
        Real error = std::fabs(calculated - value.npv); //-values[i].npv
        Real tolerance = value.tol;

        if (error>tolerance) {
            REPORT_FAILURE("value", payoffMother, payoffDaughter, exerciseMother, exerciseDaughter,
                           value.s, value.q, value.r, todaysDate, value.v, value.npv, calculated,
                           error, tolerance);
        }

        calculated = compoundOption.delta();
        error = std::fabs(calculated - value.delta);
        tolerance = value.tol;

        if (error>tolerance) {
            REPORT_FAILURE("delta", payoffMother, payoffDaughter, exerciseMother, exerciseDaughter,
                           value.s, value.q, value.r, todaysDate, value.v, value.delta, calculated,
                           error, tolerance);
        }

        calculated = compoundOption.gamma();
        error = std::fabs(calculated - value.gamma);
        tolerance = value.tol;

        if (error>tolerance) {
            REPORT_FAILURE("gamma", payoffMother, payoffDaughter, exerciseMother, exerciseDaughter,
                           value.s, value.q, value.r, todaysDate, value.v, value.gamma, calculated,
                           error, tolerance);
        }

        calculated = compoundOption.vega();
        error = std::fabs(calculated - value.vega);
        tolerance = value.tol;

        if (error>tolerance) {
            REPORT_FAILURE("vega", payoffMother, payoffDaughter, exerciseMother, exerciseDaughter,
                           value.s, value.q, value.r, todaysDate, value.v, value.vega, calculated,
                           error, tolerance);
        }

        calculated = compoundOption.theta();
        error = std::fabs(calculated - value.theta);
        tolerance = value.tol;

        if (error>tolerance) {
            REPORT_FAILURE("theta", payoffMother, payoffDaughter, exerciseMother, exerciseDaughter,
                           value.s, value.q, value.r, todaysDate, value.v, value.theta, calculated,
                           error, tolerance);
        }
    }
}

BOOST_AUTO_TEST_CASE(testTermStructures) {

    BOOST_TEST_MESSAGE("Testing compound options on non-flat curves...");

    // Reference values: nested integral of the daughter's Black price over the
    // mother's lognormal law (forward discounts and forward variance between
    // the two maturities); the closed form with correlation sqrt(V(Tm)/V(Td))
    // agrees with it for call-on-call to 1e-10.
    struct TermStructureData {
        int regime;  // 0: rates and dividends, 1: volatility, 2: all non-flat
        Option::Type typeMother;
        Option::Type typeDaughter;
        Real strikeMother;
        Real npv;
    };

    const Option::Type C = Option::Call, P = Option::Put;
    TermStructureData values[] = {
        // regime, type Mother, type Daughter, strike Mother, value
        { 0, C, C, 8.0, 6.8459930608 }, { 0, C, P, 8.0, 4.2224815702 },
        { 0, P, C, 8.0, 1.5549563665 }, { 0, P, P, 8.0, 1.5136656359 },
        { 0, C, C, 3.0, 10.3993274734 }, { 0, C, P, 3.0, 7.7827900093 },
        { 0, P, C, 3.0, 0.1827310811 }, { 0, P, P, 3.0, 0.1484143770 },
        { 1, C, C, 8.0, 10.4224177638 }, { 1, C, P, 8.0, 5.1231709955 },
        { 1, P, C, 8.0, 2.8438840841 }, { 1, P, P, 8.0, 3.1960721435 },
        { 1, C, C, 3.0, 13.1541329246 }, { 1, C, P, 3.0, 7.5902322577 },
        { 1, P, C, 3.0, 0.7233715772 }, { 1, P, P, 3.0, 0.8109057379 },
        { 2, C, C, 8.0, 8.0083817877 }, { 2, C, P, 8.0, 5.1775881959 },
        { 2, P, C, 8.0, 3.3013521514 }, { 2, P, P, 8.0, 3.0527793195 },
        { 2, C, C, 3.0, 10.4959536471 }, { 2, C, P, 3.0, 7.8021429401 },
        { 2, P, C, 3.0, 0.8633643127 }, { 2, P, P, 3.0, 0.7517743657 }
    };

    DayCounter dc = Actual365Fixed();
    Date today = Settings::instance().evaluationDate();
    Date mother = today + 365, daughter = today + 1095;
    std::vector<Date> curveDates = { today, mother, daughter };
    std::vector<Date> volDates = { mother, daughter };

    // rates/dividends are continuous zero rates, linear in time between pillars
    std::vector<Rate> flatR = { 0.03, 0.03, 0.03 }, flatQ = { 0.01, 0.01, 0.01 };
    std::vector<Rate> steepR = { 0.015, 0.015, 0.055 }, steepQ = { 0.015, 0.015, 0.045 };
    std::vector<Volatility> flatVol = { 0.2, 0.2 }, termVol = { 0.28, 0.19 };

    auto makeProcess = [&](int regime) {
        bool termRates = regime != 1;
        bool termVols = regime != 0;
        auto rTS = ext::make_shared<ZeroCurve>(curveDates, termRates ? steepR : flatR, dc);
        auto qTS = ext::make_shared<ZeroCurve>(curveDates, termRates ? steepQ : flatQ, dc);
        auto volTS = ext::make_shared<BlackVarianceCurve>(today, volDates,
                                                          termVols ? termVol : flatVol, dc);
        return ext::make_shared<BlackScholesMertonProcess>(
            Handle<Quote>(ext::make_shared<SimpleQuote>(100.0)), Handle<YieldTermStructure>(qTS),
            Handle<YieldTermStructure>(rTS), Handle<BlackVolTermStructure>(volTS));
    };

    for (const auto& value : values) {
        auto process = makeProcess(value.regime);

        CompoundOption option(ext::make_shared<PlainVanillaPayoff>(value.typeMother,
                                                                   value.strikeMother),
                              ext::make_shared<EuropeanExercise>(mother),
                              ext::make_shared<PlainVanillaPayoff>(value.typeDaughter, 100.0),
                              ext::make_shared<EuropeanExercise>(daughter));
        option.setPricingEngine(ext::make_shared<AnalyticCompoundOptionEngine>(process));

        Real calculated = option.NPV();
        if (std::fabs(calculated - value.npv) > 3.0e-4 * value.npv) {
            BOOST_ERROR("wrong compound option value on non-flat curves:"
                        << "\n    regime:     " << value.regime
                        << "\n    mother:     " << value.typeMother
                        << " (strike " << value.strikeMother << ")"
                        << "\n    daughter:   " << value.typeDaughter
                        << "\n    expected:   " << value.npv
                        << "\n    calculated: " << calculated);
        }
    }

    // equal maturities leave no variance between the two horizons
    CompoundOption degenerate(ext::make_shared<PlainVanillaPayoff>(C, 3.0),
                              ext::make_shared<EuropeanExercise>(daughter),
                              ext::make_shared<PlainVanillaPayoff>(C, 100.0),
                              ext::make_shared<EuropeanExercise>(daughter));
    degenerate.setPricingEngine(ext::make_shared<AnalyticCompoundOptionEngine>(makeProcess(2)));
    BOOST_CHECK_THROW(degenerate.NPV(), Error);
}

BOOST_AUTO_TEST_CASE(testNoCriticalSpot) {

    BOOST_TEST_MESSAGE("Testing compound options on a put when no critical spot exists...");

    // A put on the underlying is worth less than strike * Dr(Td)/Dr(Tm) at Tm, so for a
    // mother strike above that bound the mother call is worth zero and the mother put
    // is worth strike * Dr(Tm) minus the underlying put.
    Date today = Settings::instance().evaluationDate();
    DayCounter dc = Actual365Fixed();
    Date mother = today + 365, daughter = today + 1095;

    auto spot = ext::make_shared<SimpleQuote>(100.0);
    auto rRate = ext::make_shared<SimpleQuote>(0.03);
    auto qRate = ext::make_shared<SimpleQuote>(0.01);
    auto vol = ext::make_shared<SimpleQuote>(0.2);
    auto process = ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(spot),
        Handle<YieldTermStructure>(
            ext::make_shared<FlatForward>(today, Handle<Quote>(qRate), dc)),
        Handle<YieldTermStructure>(
            ext::make_shared<FlatForward>(today, Handle<Quote>(rRate), dc)),
        Handle<BlackVolTermStructure>(
            ext::make_shared<BlackConstantVol>(today, NullCalendar(), Handle<Quote>(vol), dc)));

    auto engine = ext::make_shared<AnalyticCompoundOptionEngine>(process);
    auto makeOption = [&](Option::Type motherType, Real strikeMother) {
        auto option = ext::make_shared<CompoundOption>(
            ext::make_shared<PlainVanillaPayoff>(motherType, strikeMother),
            ext::make_shared<EuropeanExercise>(mother),
            ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
            ext::make_shared<EuropeanExercise>(daughter));
        option->setPricingEngine(engine);
        return option;
    };

    // Dr(Td)/Dr(Tm) = exp(-0.06), so the bound is 94.1764...
    auto call = makeOption(Option::Call, 97.0);
    auto put = makeOption(Option::Put, 97.0);
    QL_CHECK_SMALL(call->NPV(), 1.0e-12);
    QL_CHECK_SMALL(call->delta(), 1.0e-12);
    QL_CHECK_CLOSE(put->NPV(), 83.8110706994, 1.0e-6);

    // delta, gamma and vega against bumps (the vega bump is a parallel flat-vol bump)
    Real h = 0.01, upper, lower, base = put->NPV();
    spot->setValue(100.0 + h);
    upper = put->NPV();
    spot->setValue(100.0 - h);
    lower = put->NPV();
    spot->setValue(100.0);
    QL_CHECK_CLOSE(put->delta(), (upper - lower) / (2 * h), 1.0e-6);
    QL_CHECK_CLOSE(put->gamma(), (upper - 2 * base + lower) / (h * h), 1.0e-3);
    vol->setValue(0.2 + 1.0e-5);
    upper = put->NPV();
    vol->setValue(0.2 - 1.0e-5);
    lower = put->NPV();
    vol->setValue(0.2);
    QL_CHECK_CLOSE(put->vega(), (upper - lower) / 2.0e-5, 1.0e-6);

    // every price input moves the result
    for (const auto& quote : {rRate, qRate}) {
        Real before = put->NPV();
        quote->setValue(quote->value() + 0.005);
        BOOST_CHECK(std::fabs(put->NPV() - before) > 1.0e-4);
        quote->setValue(quote->value() - 0.005);
    }

    // just below the bound the solver is used and agrees with the nested-integral value
    auto zeroCurve = [&](const std::vector<Rate>& rates) {
        return ext::make_shared<ZeroCurve>(std::vector<Date>{ today, mother, daughter }, rates,
                                           dc);
    };
    auto termProcess = ext::make_shared<BlackScholesMertonProcess>(
        Handle<Quote>(spot), Handle<YieldTermStructure>(zeroCurve({ 0.015, 0.015, 0.045 })),
        Handle<YieldTermStructure>(zeroCurve({ 0.015, 0.015, 0.055 })),
        Handle<BlackVolTermStructure>(ext::make_shared<BlackVarianceCurve>(
            today, std::vector<Date>{ mother, daughter }, std::vector<Volatility>{ 0.28, 0.19 },
            dc)));
    auto termEngine = ext::make_shared<AnalyticCompoundOptionEngine>(termProcess);
    // Kd * Dr(Td)/Dr(Tm) = 86.0708 on these curves
    const std::vector<std::pair<Option::Type, Real>> cases = {
        { Option::Put, 74.7139224127 }, { Option::Put, 83.5799298691 }, { Option::Call, 0.0 }
    };
    const std::vector<Real> strikes = { 86.0, 95.0, 95.0 };
    for (Size i = 0; i < cases.size(); ++i) {
        CompoundOption option(ext::make_shared<PlainVanillaPayoff>(cases[i].first, strikes[i]),
                              ext::make_shared<EuropeanExercise>(mother),
                              ext::make_shared<PlainVanillaPayoff>(Option::Put, 100.0),
                              ext::make_shared<EuropeanExercise>(daughter));
        option.setPricingEngine(termEngine);
        QL_CHECK_SMALL(option.NPV() - cases[i].second, 1.0e-4);
    }
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
