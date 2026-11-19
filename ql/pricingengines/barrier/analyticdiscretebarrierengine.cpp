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
#include <ql/instruments/vanillaoption.hpp>
#include <ql/pricingengines/vanilla/analyticeuropeanengine.hpp>
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

        // The last monitoring date is maturity.  An up call or a down put whose
        // strike lies beyond the barrier pays off only where the barrier has been
        // crossed at maturity, so the price is known exactly: zero for a knock-out,
        // vanilla for a knock-in.
        bool decidedAtMaturity(const BarrierOption::arguments& args,
                               const PlainVanillaPayoff& payoff) {
            bool up = (args.barrierType == Barrier::UpIn || args.barrierType == Barrier::UpOut);
            return up ? (payoff.optionType() == Option::Call && payoff.strike() >= args.barrier) :
                        (payoff.optionType() == Option::Put && payoff.strike() <= args.barrier);
        }

        Real exactValue(const BarrierOption::arguments& args,
                        const ext::shared_ptr<StrikedTypePayoff>& payoff,
                        const ext::shared_ptr<GeneralizedBlackScholesProcess>& process) {
            if (args.barrierType == Barrier::UpOut || args.barrierType == Barrier::DownOut)
                return 0.0;
            VanillaOption vanilla(payoff, args.exercise);
            vanilla.setPricingEngine(ext::make_shared<AnalyticEuropeanEngine>(process));
            return vanilla.NPV();
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
        QL_REQUIRE(payoff->strike() > 0.0, "strike must be positive");
        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "only european style option are supported");
        QL_REQUIRE(process_->x0() > 0.0, "negative or null underlying given");
        // the unshifted barrier is the one that was observed
        QL_REQUIRE(!triggered(process_->x0()), "barrier touched");

        if (decidedAtMaturity(arguments_, *payoff)) {
            results_.value = exactValue(arguments_, payoff, process_);
            return;
        }

        const Date expiry = arguments_.exercise->lastDate();
        const Time T = process_->blackVolatility()->timeFromReference(expiry);
        const Volatility sigma = process_->blackVolatility()->blackVol(expiry, payoff->strike());

        auto* continuousArgs =
            dynamic_cast<BarrierOption::arguments*>(continuousEngine_->getArguments());
        QL_REQUIRE(continuousArgs, "wrong argument type");
        *continuousArgs = arguments_;
        continuousArgs->barrier =
            shiftedBarrier(arguments_.barrierType, arguments_.barrier, sigma,
                           T / static_cast<Real>(monitoringSteps_));
        continuousEngine_->reset();
        continuousEngine_->calculate();

        const auto* continuousResults =
            dynamic_cast<const BarrierOption::results*>(continuousEngine_->getResults());
        QL_REQUIRE(continuousResults, "wrong result type");
        results_ = *continuousResults;
    }

}
