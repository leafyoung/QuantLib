/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2004, 2007 StatPro Italia srl

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

#include <ql/exercise.hpp>
#include <ql/pricingengines/blackcalculator.hpp>
#include <ql/pricingengines/cliquet/analyticcliquetengine.hpp>
#include <utility>

namespace QuantLib {

    AnalyticCliquetEngine::AnalyticCliquetEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process)
    : process_(std::move(process)) {
        registerWith(process_);
    }

    void AnalyticCliquetEngine::calculate() const {

        QL_REQUIRE(arguments_.accruedCoupon == Null<Real>() &&
                   arguments_.lastFixing == Null<Real>(),
                   "this engine cannot price options already started");
        QL_REQUIRE(arguments_.globalCap == Null<Real>() &&
                   arguments_.globalFloor == Null<Real>(),
                   "this engine cannot price globally capped/floored options");

        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "not an European option");

        ext::shared_ptr<PercentageStrikePayoff> moneyness =
            ext::dynamic_pointer_cast<PercentageStrikePayoff>(
                                                           arguments_.payoff);
        QL_REQUIRE(moneyness, "wrong payoff given");

        std::vector<Date> resetDates = arguments_.resetDates;
        resetDates.push_back(arguments_.exercise->lastDate());

        Real underlying = process_->stateVariable()->value();
        QL_REQUIRE(underlying > 0.0, "negative or null underlying");

        // The period payoff clamped between the local floor and cap is the floor itself plus a
        // long option struck at the floored level and a short one struck at the capped level.
        // A put struck at or below zero is worthless and is skipped.
        const Real floor = arguments_.localFloor == Null<Real>() ? Real(0.0) :
                                                                   arguments_.localFloor;
        const Real sgn = moneyness->optionType() == Option::Call ? 1.0 : -1.0;
        std::vector<std::pair<Real, Real>> legs(1, {moneyness->strike() + sgn * floor, 1.0});
        if (arguments_.localCap != Null<Real>())
            legs.emplace_back(moneyness->strike() + sgn * arguments_.localCap, -1.0);

        results_.value = 0.0;
        results_.delta = results_.gamma = 0.0;
        results_.theta = 0.0;
        results_.rho = results_.dividendRho = 0.0;
        results_.vega = 0.0;

        // Hold spot, calendar-date forward rates and forward variances fixed as the curve
        // reference dates advance. Each period then depends on the reference date through
        // its dividend discount factor. Use the reference-date forward rate, estimated by
        // forwardRate(d, d) over the first 1e-4 years of the curve's day counter.
        Date referenceDate = process_->dividendYield()->referenceDate();
        InterestRate q = process_->dividendYield()->forwardRate(
            referenceDate, referenceDate, process_->dividendYield()->dayCounter(), Continuous,
            NoFrequency);

        for (Size i = 1; i < resetDates.size(); i++) {

            Real weight =
                process_->dividendYield()->discount(resetDates[i-1]);
            DiscountFactor discount =
                process_->riskFreeRate()->discount(resetDates[i]) /
                process_->riskFreeRate()->discount(resetDates[i-1]);
            DiscountFactor qDiscount =
                process_->dividendYield()->discount(resetDates[i]) /
                process_->dividendYield()->discount(resetDates[i-1]);
            Real forward = underlying*qDiscount/discount;

            DayCounter rfdc  = process_->riskFreeRate()->dayCounter();
            DayCounter divdc = process_->dividendYield()->dayCounter();
            DayCounter voldc = process_->blackVolatility()->dayCounter();

            Time dt = rfdc.yearFraction(resetDates[i-1],resetDates[i]);
            Time t = divdc.yearFraction(
                                    process_->dividendYield()->referenceDate(),
                                    resetDates[i-1]);

            // the amount received whatever the path, in units of the spot at the reset date
            Real bond = weight * discount * underlying * floor;
            results_.value += bond;
            results_.delta += weight * discount * floor;
            results_.theta += q.equivalentRate(rfdc, Continuous, NoFrequency, resetDates[i - 1],
                                               resetDates[i]) * bond;
            results_.rho -= dt * bond;
            results_.dividendRho -= t * bond;

            for (const auto& leg : legs) {
                Real legStrike = leg.first;
                if (legStrike <= 0.0)
                    continue;
                Real strike = underlying * legStrike;
                auto payoff = ext::make_shared<PlainVanillaPayoff>(moneyness->optionType(),strike);
                Real variance =
                    process_->blackVolatility()->blackForwardVariance(
                                        resetDates[i-1],resetDates[i],strike);

                BlackCalculator black(payoff, forward, std::sqrt(variance), discount);
                Real w = leg.second * weight;

                results_.value += w * black.value();
                results_.delta += w * (black.delta(underlying) +
                                       legStrike * discount * black.beta());
                results_.gamma += 0.0;
                // Retain the existing per-period conversion to the risk-free day counter.
                results_.theta += q.equivalentRate(rfdc, Continuous, NoFrequency,
                                                   resetDates[i - 1], resetDates[i]) *
                                  w * black.value();

                results_.rho += w * black.rho(dt);

                Time ddt = divdc.yearFraction(resetDates[i-1],resetDates[i]);
                results_.dividendRho += w * (black.dividendRho(ddt) - t * black.value());

                ddt = voldc.yearFraction(resetDates[i-1], resetDates[i]);
                results_.vega += w * black.vega(ddt);
            }
        }

    }

}
