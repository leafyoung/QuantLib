/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2022 Skandinaviska Enskilda Banken AB (publ)

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
#include <ql/experimental/volatility/sviinterpolation.hpp>
#include <ql/experimental/volatility/svismilesection.hpp>

using namespace QuantLib;
using namespace boost::unit_test_framework;

BOOST_FIXTURE_TEST_SUITE(QuantLibTests, TopLevelFixture)

BOOST_AUTO_TEST_SUITE(SviVolatilityTests)

BOOST_AUTO_TEST_CASE(testSviSmileSection) {

    BOOST_TEST_MESSAGE("Testing SviSmileSection construction...");

    Date today = Settings::instance().evaluationDate();

    // Test time based constructor
    Time tte = 11.0 / 365;
    Real forward = 123.45;
    Real a = -0.0666;
    Real b = 0.229;
    Real sigma = 0.337;
    Real rho = 0.439;
    Real m = 0.193;
    std::vector<Real> sviParameters = {a, b, sigma, rho, m};
    // Compute the strike that yields x (log-moneyness) equal to m,
    // this simplifies the variance expression to a+b*sigma so we can test the correctness
    // against the input parameters
    Real strike = forward * std::exp(m);
    ext::shared_ptr<SviSmileSection> time_section;

    BOOST_CHECK_NO_THROW(time_section =
                             ext::make_shared<SviSmileSection>(tte, forward, sviParameters));
    BOOST_CHECK_EQUAL(time_section->atmLevel(), forward);
    QL_CHECK_CLOSE(time_section->variance(strike), a + b * sigma, 1E-10);

    // Test date based constructor
    Date date = today + Period(11, Days);
    ext::shared_ptr<SviSmileSection> date_section;

    BOOST_CHECK_NO_THROW(date_section =
                             ext::make_shared<SviSmileSection>(date, forward, sviParameters));

    BOOST_CHECK_EQUAL(date_section->atmLevel(), forward);
    QL_CHECK_CLOSE(date_section->variance(strike), a + b * sigma, 1E-10);
}

BOOST_AUTO_TEST_CASE(testSviRightWingSlope) {

    BOOST_TEST_MESSAGE("Testing SviSmileSection right wing slope bound...");

    // The right wing of the total variance has slope b(1+rho); the call price
    // vanishes at large strikes only if that slope is below 2.
    const Time tte = 1.0;
    const Real forward = 1.0;
    const Real a = 0.01, sigma = 0.1, m = 0.0;

    struct Case { Real b, rho; bool accepted; };
    const std::vector<Case> cases = {
        {1.0, 0.0, true},   // slope 1
        {1.5, 0.0, true},   // slope 1.5
        {1.0, 0.5, true},   // slope 1.5
        {2.35, -0.7, true}, // slope 0.705, b(1+|rho|) = 3.995 <= 4: only the right wing is checked
        {2.5, -0.7, false}, // slope 0.75, b(1+|rho|) = 4.25 above the bound of 4
        {2.0, 0.0, false},  // slope 2: the call price does not vanish
        {2.6, -0.2, false}, // slope 2.08, b(1+|rho|) = 3.12
        {3.0, 0.0, false},  // slope 3, b(1+|rho|) = 3
        {2.0, 0.5, false},  // slope 3
        {4.0, 0.0, false},  // slope 4
        {4.1, 0.0, false}   // b(1+|rho|) above 4
    };

    for (const auto& c : cases) {
        const std::vector<Real> params = {a, c.b, sigma, c.rho, m};
        if (c.accepted) {
            BOOST_CHECK_NO_THROW(SviSmileSection(tte, forward, params));
        } else {
            BOOST_CHECK_THROW(SviSmileSection(tte, forward, params), Error);
        }
    }
}

BOOST_AUTO_TEST_CASE(testSviCallPriceTail) {

    BOOST_TEST_MESSAGE("Testing SviSmileSection call price at large strikes...");

    // Reference call prices (discount 1) from an independent evaluation of
    // C = N(d1) - K N(d2) with the SVI total variance at k = log(K/F).
    const Time tte = 1.0;
    const Real forward = 1.0;
    const Real a = 0.01, sigma = 0.1, m = 0.0;

    struct Case { Real b, rho, call8, call12; };
    const std::vector<Case> cases = {
        {1.0, 0.0, 0.0460002960118183, 0.0252096521658758},
        {1.5, 0.0, 0.203208556515268, 0.179667838163160},
        {1.0, 0.5, 0.203193316009886, 0.179660204409079},
        {1.0, -0.5, 0.000507283716789485, 0.0000458947549149203}
    };

    for (const auto& c : cases) {
        SviSmileSection section(tte, forward, {a, c.b, sigma, c.rho, m});
        const Real low = section.optionPrice(forward * std::exp(8.0), Option::Call);
        const Real high = section.optionPrice(forward * std::exp(12.0), Option::Call);
        QL_CHECK_CLOSE(low, c.call8, 1.0e-4);
        QL_CHECK_CLOSE(high, c.call12, 1.0e-4);
        // an arbitrage-free call price decreases in the strike
        if (!(high < low))
            BOOST_ERROR("call price does not decrease at large strikes (b=" << c.b
                        << ", rho=" << c.rho << "): " << low << " -> " << high);
    }

    // The wing slope b(1+rho) = 3 makes the call price rise towards the forward;
    // such a smile is rejected at construction.
    BOOST_CHECK_THROW(SviSmileSection(tte, forward, {a, 3.0, sigma, 0.0, m}), Error);
}

BOOST_AUTO_TEST_CASE(testSviInterpolationWingSlope) {

    BOOST_TEST_MESSAGE("Testing SviInterpolation calibration under the wing slope bound...");

    const Time tte = 1.0;
    const Real forward = 1.0;
    const std::vector<Real> source = {0.01, 1.0, 0.1, 0.5, 0.0}; // slope 1.5
    SviSmileSection reference(tte, forward, source);

    std::vector<Real> strikes, vols;
    for (Real k = -1.5; k <= 1.5001; k += 0.25) {
        strikes.push_back(forward * std::exp(k));
        vols.push_back(reference.volatility(strikes.back()));
    }

    SviInterpolation fit(strikes.begin(), strikes.end(), vols.begin(), tte, forward,
                         Null<Real>(), Null<Real>(), Null<Real>(), Null<Real>(),
                         Null<Real>(), false, false, false, false, false, true);
    fit.update();
    QL_CHECK_SMALL(fit.rmsError(), 1.0e-4);
    BOOST_CHECK_MESSAGE(fit.b() * (1.0 + fit.rho()) < 2.0,
                        "calibrated right wing slope " << fit.b() * (1.0 + fit.rho())
                        << " is not below 2");

    // With rho fixed at a positive value the default seed for b must also satisfy the bound.
    SviInterpolation fixedRho(strikes.begin(), strikes.end(), vols.begin(), tte, forward,
                              Null<Real>(), Null<Real>(), Null<Real>(), 0.5, Null<Real>(),
                              false, false, false, true, false, true);
    fixedRho.update();
    QL_CHECK_SMALL(fixedRho.rmsError(), 1.0e-4);
    BOOST_CHECK_MESSAGE(fixedRho.b() * (1.0 + fixedRho.rho()) < 2.0,
                        "calibrated right wing slope with fixed rho is not below 2");

    // With b fixed and rho free, the admissible rho depends on b.
    std::vector<Real> fixedBSource = {0.01, 0.9, 0.1, 0.95, 0.2};
    SviSmileSection steep(tte, forward, fixedBSource);
    std::vector<Real> steepVols;
    for (Real strike : strikes)
        steepVols.push_back(steep.volatility(strike));
    SviInterpolation fixedB(strikes.begin(), strikes.end(), steepVols.begin(), tte, forward,
                            Null<Real>(), 1.5, Null<Real>(), Null<Real>(), Null<Real>(),
                            false, true, false, false, false, true);
    BOOST_CHECK_NO_THROW(fixedB.update());
    QL_CHECK_CLOSE(fixedB.b(), 1.5, 1.0e-12);
    BOOST_CHECK_MESSAGE(fixedB.b() * (1.0 + fixedB.rho()) < 2.0,
                        "calibrated right wing slope with fixed b is not below 2");
    BOOST_CHECK_MESSAGE(fixedB.rho() > 0.25,
                        "calibrated rho " << fixedB.rho() << " should lie near its upper bound");

    // With b and rho both fixed, rho is kept as given, also close to the admissible edge.
    SviInterpolation bothFixed(strikes.begin(), strikes.end(), vols.begin(), tte, forward,
                               Null<Real>(), 1.5, Null<Real>(), 0.3333333, Null<Real>(),
                               false, true, false, true, false, true);
    BOOST_CHECK_NO_THROW(bothFixed.update());
    QL_CHECK_CLOSE(bothFixed.rho(), 0.3333333, 1.0e-10);
    QL_CHECK_CLOSE(bothFixed.b(), 1.5, 1.0e-12);

    // A fixed rho beyond the range of the transformation is not replaced by another value.
    SviInterpolation edgeRho(strikes.begin(), strikes.end(), vols.begin(), tte, forward,
                             Null<Real>(), Null<Real>(), Null<Real>(), 0.9999995, Null<Real>(),
                             false, false, false, true, false, true);
    BOOST_CHECK_THROW(edgeRho.update(), Error);

    // A fixed pair with right wing slope above 2 is rejected.
    BOOST_CHECK_THROW(SviInterpolation(strikes.begin(), strikes.end(), vols.begin(), tte,
                                       forward, Null<Real>(), 1.5, Null<Real>(), 0.5,
                                       Null<Real>(), false, true, false, true, false, true),
                      Error);
}

BOOST_AUTO_TEST_CASE(testSviTransformationRange) {

    BOOST_TEST_MESSAGE("Testing SviSpecs parameter transformation range...");

    detail::SviSpecs specs;
    const std::vector<bool> isFixed(5, false);
    const std::vector<Real> params(5, Null<Real>());
    const Real forward = 1.0;
    const Time tte = 1.0;

    std::vector<Real> angles = {-M_PI / 2, M_PI / 2};
    for (Real x3 = -1.5; x3 <= 1.5; x3 += 0.1)
        angles.push_back(x3);

    // Every parameter set the transformation can produce is admissible.
    for (Real x3 : angles) {
        for (Real x1 : std::vector<Real>{-1.0e6, -10.0, 0.0, 10.0, 1.0e6}) {
            const Array x = {0.3, x1, 0.2, x3, 0.0};
            const Array y = specs.direct(x, isFixed, params, forward);
            BOOST_CHECK_NO_THROW(detail::checkSviParameters(y[0], y[1], y[2], y[3], y[4], tte));
        }
    }

    // The random guesses used for restarts are admissible as well.
    for (Real r = 0.05; r < 1.0; r += 0.1) {
        for (Real r3 = 0.0; r3 <= 1.0; r3 += 0.1) {
            Array values(5);
            const std::vector<Real> seq = {r, r3, r, r, r};
            specs.guess(values, isFixed, forward, tte, seq, std::vector<Real>());
            BOOST_CHECK_NO_THROW(detail::checkSviParameters(values[0], values[1], values[2],
                                                            values[3], values[4], tte));
        }
    }

    // The same holds with b fixed and rho free.
    for (Real fixedB : std::vector<Real>{0.5, 1.0, 1.5, 2.0, 2.5}) {
        const std::vector<bool> bFixed = {false, true, false, false, false};
        const std::vector<Real> bParams = {Null<Real>(), fixedB, Null<Real>(), Null<Real>(),
                                           Null<Real>()};
        for (Real x3 : angles) {
            const Array x = {0.3, 0.0, 0.2, x3, 0.0};
            const Array y = specs.direct(x, bFixed, bParams, forward);
            BOOST_CHECK_NO_THROW(detail::checkSviParameters(y[0], y[1], y[2], y[3], y[4], tte));
        }
        for (Real r3 = 0.0; r3 <= 1.0; r3 += 0.1) {
            Array values = {0.0, fixedB, 0.0, 0.0, 0.0};
            const std::vector<Real> seq = {0.5, r3, 0.5, 0.5};
            specs.guess(values, bFixed, forward, tte, seq, std::vector<Real>());
            BOOST_CHECK_NO_THROW(detail::checkSviParameters(values[0], values[1], values[2],
                                                            values[3], values[4], tte));
        }
    }

    // The default rho lies inside the interval admitted for a fixed b, and equals -0.4 when
    // that is well inside it.
    for (Real fixedB : std::vector<Real>{1.5, 2.9}) {
        std::vector<Real> seedParams = {Null<Real>(), fixedB, Null<Real>(), Null<Real>(),
                                        Null<Real>()};
        std::vector<bool> seedFixed = {false, true, false, false, false};
        specs.defaultValues(seedParams, seedFixed, forward, tte, std::vector<Real>());
        BOOST_CHECK_NO_THROW(detail::checkSviParameters(seedParams[0], seedParams[1],
                                                        seedParams[2], seedParams[3],
                                                        seedParams[4], tte));
        if (fixedB == 1.5)
            QL_CHECK_CLOSE(seedParams[3], -0.4, 1.0e-12);
        else
            BOOST_CHECK_MESSAGE(seedParams[3] > -0.379 && seedParams[3] < -0.310,
                                "default rho " << seedParams[3]
                                               << " outside the interval for b = 2.9");
    }

    // inverse undoes direct for b.
    const Array y = {0.02, 0.8, 0.2, 0.5, 0.1};
    const Array x = specs.inverse(y, isFixed, params, forward);
    const Array roundTrip = specs.direct(x, isFixed, params, forward);
    QL_CHECK_CLOSE(roundTrip[1], y[1], 1.0e-8);

    // inverse undoes direct for rho when b is fixed.
    const std::vector<bool> bFixed = {false, true, false, false, false};
    const std::vector<Real> bParams = {Null<Real>(), 1.5, Null<Real>(), Null<Real>(), Null<Real>()};
    const Array yFixed = {0.02, 1.5, 0.2, 0.2, 0.1};
    const Array xFixed = specs.inverse(yFixed, bFixed, bParams, forward);
    const Array roundTripFixed = specs.direct(xFixed, bFixed, bParams, forward);
    QL_CHECK_CLOSE(roundTripFixed[3], yFixed[3], 1.0e-8);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE_END()
