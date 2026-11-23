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
#include <ql/math/distributions/normaldistribution.hpp>
#include <ql/math/solvers1d/brent.hpp>
#include <ql/pricingengines/asian/analytic_discr_arith_av_strike_bound.hpp>
#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace QuantLib {

    namespace {

        struct UndiscountedBounds {
            Real call;
            Real put;
        };

        // Lower bounds E[(E[S_T - A | Z])^+] and E[(E[A - S_T | Z])^+] for
        // Z = ln S_T - mean(ln S_{t_i}) under Black-Scholes with the given forwards.
        // Given Z = z, E[S_T - A | Z = z] = sum_k a_k F_k exp(c_k z - c_k^2/2), with
        // a = (-1/N, ..., -1/N, 1) and c_k = Cov(ln S_k, Z)/sd(Z) nondecreasing in k.
        // The coefficients change sign once when ordered by c_k, so the function
        // has a single zero z* (it is negative at -infinity, since the first fixing
        // has c = 0, and positive at +infinity), and the bounds are normal sums at z*.
        UndiscountedBounds conditioningBounds(const std::vector<Real>& forwards,
                                              const std::vector<Time>& times,
                                              Real terminalForward,
                                              Time maturity,
                                              Real variance) {
            Size n = forwards.size();
            Real count = static_cast<Real>(n);

            // Mean of the fixing forwards, and the terms ordered as fixings, then S_T.
            Real averageForward = 0.0;
            for (Real forward : forwards)
                averageForward += forward / count;
            Real intrinsic = terminalForward - averageForward;

            // Weight i/N applies to the Brownian increment ending at t_i, weight 1 to
            // the one ending at T. Cumulating these nonnegative pieces gives
            // Cov(ln S_k, Z)/sigma^2 and Var(Z)/sigma^2 without cancellation.
            std::vector<Real> covariance(n + 1);
            Real varianceSum = 0.0;
            Real covarianceSum = 0.0;
            Time previous = 0.0;
            for (Size i = 0; i < n; ++i) {
                Real weight = static_cast<Real>(i) / count;
                Time step = times[i] - previous;
                covarianceSum += weight * step;
                varianceSum += weight * weight * step;
                covariance[i] = covarianceSum;
                previous = times[i];
            }
            covariance[n] = covarianceSum + (maturity - previous);
            varianceSum += maturity - previous;

            Real sdSquared = variance * varianceSum;
            // No randomness left in S_T - A (zero volatility, or all fixings at T).
            if (sdSquared == 0.0)
                return {std::max<Real>(intrinsic, 0.0), std::max<Real>(-intrinsic, 0.0)};

            Real sd = std::sqrt(sdSquared);
            std::vector<Real> c(n + 1), sign(n + 1), logSize(n + 1), size(n + 1);
            for (Size k = 0; k <= n; ++k) {
                c[k] = variance * covariance[k] / sd;
                Real forward = (k < n ? Real(forwards[k] / count) : Real(terminalForward));
                sign[k] = (k < n ? -1.0 : 1.0);
                size[k] = forward;
                logSize[k] = std::log(forward) - c[k] * c[k] / 2.0;
            }

            // Beyond this the exponents of g lose the fixing weights to rounding.
            QL_REQUIRE(c[n] <= 1000.0,
                       "volatility too large: standard deviation " << c[n]
                                                                   << " to the exercise");

            // g(z) scaled by its largest term: finite and sign-preserving for any z.
            auto scaledG = [&](Real z) -> Real {
                Real top = logSize[0] + c[0] * z;
                for (Size k = 1; k <= n; ++k)
                    top = std::max<Real>(top, logSize[k] + c[k] * z);
                Real sum = 0.0;
                for (Size k = 0; k <= n; ++k)
                    sum += sign[k] * std::exp(logSize[k] + c[k] * z - top);
                return sum;
            };

            // Beyond this |z| every normal value below is 0 or 1 in double precision,
            // so the root is replaced by the end of the bracket when it lies outside.
            constexpr double saturation = 40.0;
            Real limit = saturation + c[n];
            Real root;
            if (scaledG(limit) <= 0.0) {
                root = limit;
            } else if (scaledG(-limit) >= 0.0) {
                root = -limit;
            } else {
                Brent solver;
                solver.setMaxEvaluations(200);
                root = solver.solve(scaledG, 1.0e-14, 0.0, -limit, limit);
                QL_ENSURE(std::fabs(scaledG(root)) < 1.0e-8,
                          "conditioning root not converged: g(" << root << ") = " << scaledG(root));
            }

            CumulativeNormalDistribution cdf;
            Real call = 0.0;
            Real put = 0.0;
            for (Size k = 0; k <= n; ++k) {
                call += sign[k] * size[k] * cdf(c[k] - root);
                put -= sign[k] * size[k] * cdf(root - c[k]);
            }
            return {std::max<Real>(call, 0.0), std::max<Real>(put, 0.0)};
        }

    }

    AnalyticDiscreteArithmeticAverageStrikeBoundEngine::
        AnalyticDiscreteArithmeticAverageStrikeBoundEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process)
    : process_(std::move(process)) {
        registerWith(process_);
    }

    void AnalyticDiscreteArithmeticAverageStrikeBoundEngine::calculate() const {

        QL_REQUIRE(arguments_.averageType == Average::Arithmetic,
                   "not an arithmetic average option");

        QL_REQUIRE(arguments_.exercise->type() == Exercise::European, "not an European option");

        QL_REQUIRE(arguments_.pastFixings == 0, "past fixings currently not managed");

        ext::shared_ptr<PlainVanillaPayoff> payoff =
            ext::dynamic_pointer_cast<PlainVanillaPayoff>(arguments_.payoff);
        QL_REQUIRE(payoff, "non-plain payoff given");

        Real underlying = process_->stateVariable()->value();
        QL_REQUIRE(underlying > 0.0, "positive underlying value required");

        Date referenceDate = process_->riskFreeRate()->referenceDate();
        Date volReferenceDate = process_->blackVolatility()->referenceDate();
        DayCounter voldc = process_->blackVolatility()->dayCounter();

        std::vector<Time> fixingTimes;
        std::vector<Real> fixingForwards;
        for (const Date& fixingDate : arguments_.fixingDates) {
            if (fixingDate >= referenceDate) {
                Time t = voldc.yearFraction(volReferenceDate, fixingDate);
                QL_REQUIRE(t >= 0.0, "fixing date " << fixingDate
                                                    << " precedes the volatility reference date");
                fixingTimes.push_back(t);
                fixingForwards.push_back(underlying *
                                         process_->dividendYield()->discount(fixingDate) /
                                         process_->riskFreeRate()->discount(fixingDate));
            }
        }
        QL_REQUIRE(!fixingTimes.empty(), "no future fixings");

        Date exDate = arguments_.exercise->lastDate();
        Time residualTime = voldc.yearFraction(volReferenceDate, exDate);
        QL_REQUIRE(fixingTimes.back() <= residualTime,
                   "fixing dates after the exercise date are not allowed");

        Volatility volatility = process_->blackVolatility()->blackVol(exDate, underlying);
        QL_REQUIRE(volatility >= 0.0 && std::isfinite(volatility),
                   "non-negative finite volatility required: " << volatility);

        DiscountFactor dividendDiscount = process_->dividendYield()->discount(exDate);
        DiscountFactor riskFreeDiscount = process_->riskFreeRate()->discount(exDate);
        Real forward = underlying * dividendDiscount / riskFreeDiscount;
        QL_REQUIRE(forward > 0.0 && riskFreeDiscount > 0.0,
                   "positive forward and discount factor required");
        for (Real fixingForward : fixingForwards)
            QL_REQUIRE(fixingForward > 0.0, "positive fixing forwards required");

        UndiscountedBounds bounds = conditioningBounds(fixingForwards, fixingTimes, forward,
                                                       residualTime, volatility * volatility);
        results_.value =
            riskFreeDiscount * (payoff->optionType() == Option::Call ? bounds.call : bounds.put);
    }

}
