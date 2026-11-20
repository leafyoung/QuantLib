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

/*! \file analyticdiscretebarrierengine.hpp
    \brief Barrier option engine with discrete monitoring (continuity correction)
*/

#ifndef quantlib_analytic_discrete_barrier_engine_hpp
#define quantlib_analytic_discrete_barrier_engine_hpp

#include <ql/pricingengines/barrier/analyticbarrierengine.hpp>

namespace QuantLib {

    //! Barrier option engine for discretely monitored barriers
    /*! Applies the continuity correction of Broadie, Glasserman and Kou,
        "A Continuity Correction for Discrete Barrier Options",
        Mathematical Finance 7(4), 1997, pp. 325-349: the price is that of
        the continuously monitored option (AnalyticBarrierEngine) with the
        barrier \f$ H \f$ moved away from the spot to
        \f$ H e^{\pm\beta\sigma\sqrt{\Delta t}} \f$ (up barriers +, down
        barriers -), where \f$ \beta = -\zeta(1/2)/\sqrt{2\pi} \approx 0.5826 \f$.

        The barrier is observed at \f$ N \f$ equally spaced dates
        \f$ i \Delta t \f$, \f$ i = 1, \ldots, N \f$, \f$ \Delta t = T/N \f$,
        the first one step after the reference date of the volatility term
        structure and the last at maturity.  Only European options without
        rebate are supported.  \f$ \sigma \f$ is the Black volatility at the
        expiry and the strike and \f$ T \f$ the time to expiry on the clock
        of the volatility surface; the result is stated for a constant
        volatility.  When the strike lies beyond the barrier (up call with
        \f$ K \ge H \f$, down put with \f$ K \le H \f$) the last monitoring
        date decides the payoff and the result is exact: zero for a
        knock-out, the vanilla price for a knock-in.

        \warning The correction is asymptotic.  Reverse barriers (up calls,
                 down puts) are much less accurate at coarse monitoring; a
                 knock-in carries the opposite-signed error of the matching
                 knock-out.  The price is discontinuous in the strike at
                 \f$ K = H \f$.  With zero volatility and the spot on the
                 barrier, the convention of AnalyticBarrierEngine applies.

        \ingroup barrierengines

        \test the correctness of the returned value is tested in
              DiscreteBarrierOptionTests
    */
    class AnalyticDiscreteBarrierEngine : public BarrierOption::engine {
      public:
        AnalyticDiscreteBarrierEngine(ext::shared_ptr<GeneralizedBlackScholesProcess> process,
                                      Size monitoringSteps);
        void calculate() const override;

      private:
        ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
        Size monitoringSteps_;
        ext::shared_ptr<AnalyticBarrierEngine> continuousEngine_;
    };

}

#endif
