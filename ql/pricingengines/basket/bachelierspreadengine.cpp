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

#include <ql/exercise.hpp>
#include <ql/math/functional.hpp>
#include <ql/pricingengines/basket/bachelierspreadengine.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <algorithm>
#include <utility>

namespace QuantLib {

    BachelierSpreadEngine::BachelierSpreadEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process1,
        ext::shared_ptr<GeneralizedBlackScholesProcess> process2,
        Handle<Quote> absoluteVolatility1,
        Handle<Quote> absoluteVolatility2,
        Real correlation)
    : process1_(std::move(process1)), process2_(std::move(process2)),
      vol1_(std::move(absoluteVolatility1)), vol2_(std::move(absoluteVolatility2)),
      rho_(correlation) {
        QL_REQUIRE(rho_ >= -1.0 && rho_ <= 1.0,
                   "correlation (" << rho_ << ") must be in [-1, 1]");
        registerWith(process1_);
        registerWith(process2_);
        registerWith(vol1_);
        registerWith(vol2_);
    }

    void BachelierSpreadEngine::calculate() const {
        auto exercise = ext::dynamic_pointer_cast<EuropeanExercise>(arguments_.exercise);
        QL_REQUIRE(exercise, "not an European exercise");

        auto spreadPayoff = ext::dynamic_pointer_cast<SpreadBasketPayoff>(arguments_.payoff);
        QL_REQUIRE(spreadPayoff, "spread payoff expected");

        auto payoff = ext::dynamic_pointer_cast<PlainVanillaPayoff>(spreadPayoff->basePayoff());
        QL_REQUIRE(payoff, "non-plain payoff given");

        const Date maturity = exercise->lastDate();
        auto forward = [&maturity](const ext::shared_ptr<GeneralizedBlackScholesProcess>& p) {
            return p->stateVariable()->value() / p->riskFreeRate()->discount(maturity) *
                   p->dividendYield()->discount(maturity);
        };

        const Real sigma1 = vol1_->value();
        const Real sigma2 = vol2_->value();
        QL_REQUIRE(sigma1 >= 0.0 && sigma2 >= 0.0, "absolute volatilities must be non-negative");

        const Time t = process1_->time(maturity);
        // the variance is nonnegative in exact arithmetic (it equals
        // (sigma1 - sigma2)^2 + 2*(1 - rho)*sigma1*sigma2); the floor only
        // guards rounding when rho = 1 and sigma1 = sigma2.
        const Real variance =
            std::max(0.0, squared(sigma1) + squared(sigma2) - 2.0 * rho_ * sigma1 * sigma2) * t;

        results_.value = bachelierBlackFormula(payoff->optionType(), payoff->strike(),
                                               forward(process1_) - forward(process2_),
                                               std::sqrt(variance),
                                               process1_->riskFreeRate()->discount(maturity));
    }
}
