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

/*! \file analytic_discr_arith_av_strike_bound.hpp
    \brief Conditioning lower bound for discrete arithmetic average-strike Asian options
*/

#ifndef quantlib_analytic_discrete_arithmetic_average_strike_bound_asian_engine_hpp
#define quantlib_analytic_discrete_arithmetic_average_strike_bound_asian_engine_hpp

#include <ql/instruments/asianoption.hpp>
#include <ql/processes/blackscholesprocess.hpp>

namespace QuantLib {

    //! Lower bound for European discrete arithmetic average-strike Asian options
    /*! This engine returns a <b>lower bound</b> of the price of a discrete
        arithmetic average-strike Asian option with European exercise, with
        payoff \f$ (S_T - A)^+ \f$ for a call and \f$ (A - S_T)^+ \f$ for a
        put, where \f$ A \f$ is the arithmetic mean of the underlying at the
        fixing dates.  It is <b>not</b> the exact price.

        The bound is obtained by Jensen conditioning on a Gaussian variable,
        after the conditioning idea of L. C. G. Rogers and Z. Shi, "The
        value of an Asian option", J. Appl. Probab. 32 (1995), 1077-1088,
        as presented for the continuous floating-strike case in G. W.
        Thompson, "Fast narrow bounds on the value of Asian options",
        Judge Institute of Management working paper wp0209; the discrete
        form implemented here is a transcription of that conditioning.
        With \f$ Z = \ln S_T - \frac1N \sum_i \ln
        S_{t_i} \f$, which is Gaussian under the Black-Scholes model,
        \f$ E[(S_T - A)^+] \ge E[(E[S_T - A \mid Z])^+] \f$.  Given
        \f$ Z \f$ every \f$ S_{t_i} \f$ is lognormal, so
        \f$ g(z) = E[S_T - A \mid Z = z] \f$ is a signed sum of
        exponentials with exactly one sign change; the bound is then
        a sum of cumulative normal values at the root of \f$ g \f$,
        found numerically.  The put value is the corresponding bound for
        the put payoff; the call and put bounds are computed from the same root and satisfy
        put-call parity to rounding error.
        Only the exact Black-Scholes moments of the underlying are used.
        Term structures of rates and dividends are allowed; the
        volatility is the constant Black volatility at the exercise date,
        as in AnalyticDiscreteGeometricAverageStrikeAsianEngine.

        When no randomness is left in \f$ S_T - A \f$ (zero volatility,
        or every fixing at the exercise date) the value is the discounted
        deterministic payoff.

        The payoff must be a PlainVanillaPayoff; as in the other
        average-strike engines only its option type is used, the strike
        is ignored.  Past fixings are not managed and the running
        accumulator is ignored.  Fixing dates before the reference date of
        the risk-free curve are ignored, as in the geometric engine.

        The bound is low-biased by construction.  Measured against a
        Monte Carlo reference (MCDiscreteArithmeticASEngine, Sobol
        sequences with Brownian bridge, 2^20-1 samples; S=97, r=5%, q=2%,
        T=1; 1, 4, 10 and 52 equally spaced fixings; the single fixing is
        at t=0.5; calls and puts), the shortfall of the bound is
        - at most 0.012% at volatility 20%,
        - at most 0.044% at volatility 40%,
        - at most 0.172% at volatility 80% (put, four fixings).
        The bound is exact for a
        single fixing.  The sign of the deviation is negative apart from
        Monte Carlo noise (a lower bound cannot exceed the price).  No
        accuracy is guaranteed beyond this grid.

        \test
        - the bound is reproduced by closed-form forward-start values for
          a single fixing, by deterministic limits, and by put-call
          parity;
        - the bound is checked against stored Monte Carlo reference values
          for the grid above: it must not exceed them by more than their
          noise, nor fall short by more than the tolerances 0.02%, 0.06%
          and 0.25% for volatilities 20%, 40% and 80%.

        \ingroup asianengines
    */
    class AnalyticDiscreteArithmeticAverageStrikeBoundEngine
    : public DiscreteAveragingAsianOption::engine {
      public:
        explicit AnalyticDiscreteArithmeticAverageStrikeBoundEngine(
            ext::shared_ptr<GeneralizedBlackScholesProcess> process);
        void calculate() const override;

      private:
        ext::shared_ptr<GeneralizedBlackScholesProcess> process_;
    };

}

#endif
