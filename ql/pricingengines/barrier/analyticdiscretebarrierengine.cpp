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

#include <ql/pricingengines/barrier/analyticdiscretebarrierengine.hpp>
#include <ql/exercise.hpp>
#include <cmath>
#include <utility>

namespace QuantLib {

    namespace {

        // -zeta(1/2)/sqrt(2 pi)
        constexpr double broadieGlassermanKouBeta = 0.5825971579390106;

        // The barrier moved away from the spot by exp(beta sigma sqrt(dt)).
        Real shiftedBarrier(Barrier::Type type, Real barrier, Volatility sigma, Time dt) {
            Real shift = std::exp(broadieGlassermanKouBeta * sigma * std::sqrt(dt));
            bool down = (type == Barrier::DownIn || type == Barrier::DownOut);
            return down ? Real(barrier / shift) : Real(barrier * shift);
        }

    }

    AnalyticDiscreteBarrierEngine::AnalyticDiscreteBarrierEngine(
        ext::shared_ptr<GeneralizedBlackScholesProcess> process, Size monitoringSteps)
    : process_(std::move(process)), monitoringSteps_(monitoringSteps),
      continuousEngine_(ext::make_shared<AnalyticBarrierEngine>(process_)) {
        QL_REQUIRE(monitoringSteps_ >= 1, "at least one monitoring date is required");
        registerWith(process_);
    }

    void AnalyticDiscreteBarrierEngine::calculate() const {

        QL_REQUIRE(arguments_.rebate == 0.0,
                   "rebates are not supported by the discrete-monitoring correction");

        ext::shared_ptr<PlainVanillaPayoff> payoff =
            ext::dynamic_pointer_cast<PlainVanillaPayoff>(arguments_.payoff);
        QL_REQUIRE(payoff, "non-plain payoff given");
        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "only european style option are supported");

        // the unshifted barrier is the one that was observed
        QL_REQUIRE(!triggered(process_->x0()), "barrier touched");

        Date expiry = arguments_.exercise->lastDate();
        Time T = process_->blackVolatility()->timeFromReference(expiry);
        Volatility sigma = process_->blackVolatility()->blackVol(expiry, payoff->strike());

        auto* continuousArgs =
            dynamic_cast<BarrierOption::arguments*>(continuousEngine_->getArguments());
        QL_REQUIRE(continuousArgs, "wrong argument type");
        *continuousArgs = arguments_;
        continuousArgs->barrier =
            shiftedBarrier(arguments_.barrierType, arguments_.barrier, sigma,
                           T / static_cast<Real>(monitoringSteps_));
        continuousEngine_->calculate();

        auto* continuousResults =
            dynamic_cast<const BarrierOption::results*>(continuousEngine_->getResults());
        QL_REQUIRE(continuousResults, "wrong result type");
        results_ = *continuousResults;
    }

}
