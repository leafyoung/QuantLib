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
#include <ql/pricingengines/cliquet/analyticperformanceengine.hpp>
#include <utility>

namespace QuantLib {

    AnalyticPerformanceEngine::AnalyticPerformanceEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process)
    : process_(std::move(process)) {
        registerWith(process_);
    }

    void AnalyticPerformanceEngine::calculate() const {

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

        auto payoff = ext::make_shared<PlainVanillaPayoff>(moneyness->optionType(), 1.0);

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
        // its risk-free discount factor. Use the reference-date forward rate, estimated by
        // forwardRate(d, d) over the first 1e-4 years of the curve's day counter.
        Date referenceDate = process_->riskFreeRate()->referenceDate();
        Rate r = process_->riskFreeRate()->forwardRate(referenceDate, referenceDate,
                                                       process_->riskFreeRate()->dayCounter(),
                                                       Continuous, NoFrequency);

        for (Size i = 1; i < resetDates.size(); i++) {

            DiscountFactor discount =
                process_->riskFreeRate()->discount(resetDates[i-1]);
            DiscountFactor rDiscount =
                process_->riskFreeRate()->discount(resetDates[i]) /
                process_->riskFreeRate()->discount(resetDates[i-1]);
            DiscountFactor qDiscount =
                process_->dividendYield()->discount(resetDates[i]) /
                process_->dividendYield()->discount(resetDates[i-1]);
            DayCounter rfdc  = process_->riskFreeRate()->dayCounter();
            DayCounter divdc = process_->dividendYield()->dayCounter();
            DayCounter voldc = process_->blackVolatility()->dayCounter();

            Time dt = rfdc.yearFraction(resetDates[i-1],resetDates[i]);
            Time t = rfdc.yearFraction(
                                  process_->riskFreeRate()->referenceDate(),
                                  resetDates[i-1]);

            // the amount received whatever the path
            Real bond = discount * rDiscount * floor;
            results_.value += bond;
            results_.theta += r * bond;
            results_.rho -= (t + dt) * bond;

            for (const auto& leg : legs) {
                Real strike = leg.first;
                if (strike <= 0.0)
                    continue;
                Real forward = (1.0/strike)*qDiscount/rDiscount;
                Real variance =
                    process_->blackVolatility()->blackForwardVariance(
                                        resetDates[i-1],resetDates[i],
                                        underlying * strike);

                BlackCalculator black(payoff, forward, std::sqrt(variance), rDiscount);
                Real weight = leg.second * discount * strike;

                results_.value += weight * black.value();
                results_.delta += 0.0;
                results_.gamma += 0.0;
                results_.theta += r * weight * black.value();

                results_.rho += weight * (black.rho(dt) - t * black.value());

                Time ddt = divdc.yearFraction(resetDates[i-1],resetDates[i]);
                results_.dividendRho += weight * black.dividendRho(ddt);

                ddt = voldc.yearFraction(resetDates[i-1], resetDates[i]);
                results_.vega += weight * black.vega(ddt);
            }
        }

    }

}
