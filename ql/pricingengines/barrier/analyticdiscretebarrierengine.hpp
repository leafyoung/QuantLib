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
    /*! Prices a barrier option whose barrier is observed only at
        \f$ N \f$ equally spaced dates, using the continuity correction of
        Broadie, Glasserman and Kou, "A Continuity Correction for Discrete
        Barrier Options", Mathematical Finance 7(4), 1997, pp. 325-349.

        The monitoring dates are \f$ t_i = i \Delta t \f$ for
        \f$ i = 1, \ldots, N \f$ with \f$ \Delta t = T/N \f$, so that the
        first date is one step after the valuation date and the last one
        coincides with maturity.  The price is that of the continuously
        monitored option (AnalyticBarrierEngine) with the barrier \f$ H \f$
        moved away from the spot to
        \f[
          H e^{+\beta\sigma\sqrt{\Delta t}} \ \text{(up barriers)}, \qquad
          H e^{-\beta\sigma\sqrt{\Delta t}} \ \text{(down barriers)},
        \f]
        where \f$ \beta = -\zeta(1/2)/\sqrt{2\pi} \approx 0.5826 \f$ and
        \f$ \zeta \f$ is the Riemann zeta function.  A knock-out is therefore
        worth more, and a knock-in less, than its continuously monitored
        counterpart, as crossings between monitoring dates are not seen.

        The correction is an asymptotic approximation for small
        \f$ \sigma\sqrt{\Delta t} \f$; its accuracy degrades for large
        \f$ \Delta t \f$ and for spot levels close to the barrier.  It is
        derived for a constant volatility.  \f$ \sigma \f$ is read from the
        Black volatility surface of the process at the expiry and the
        option strike; with a non-flat surface this is an approximation
        accepted by the user.

        Only options without rebate are supported.

        \ingroup barrierengines

        \test the correctness of the returned value is tested against
              prices at the shifted barrier, an independent Monte Carlo
              simulation of the discretely monitored process, and the
              convergence to the continuous price.
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
