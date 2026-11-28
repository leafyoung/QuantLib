/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2018 Klaus Spanderen

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

/*! \file analyticcevengine.hpp
    \brief  Pricing engine for European vanilla options using a
    constant elasticity of variance (CEV) model
*/

#ifndef quantlib_analytic_cev_engine_hpp
#define quantlib_analytic_cev_engine_hpp

#include <ql/instruments/vanillaoption.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>

namespace QuantLib {

    //! constant elasticity of variance process (absorbing boundary at f=0)
    /*! \f[
         df_t = \alpha f_t^\beta \mathrm{d}W_t
        \f]
    */

    /*! References:

        D.R. Brecher, A.E. Lindsay, Results on the CEV Process, Past and Present
        https://www.fincad.com/sites/default/files/wysiwyg/Resources-Wiki/cev-process-working-paper.pdf
    */

    class CEVCalculator {
      public:
        CEVCalculator(Real f0, Real alpha, Real beta);

        Real value(Option::Type optionType, Real strike, Time t) const;

        Real f0()    const { return f0_; }
        Real alpha() const { return alpha_; }
        Real beta()  const { return beta_; }

      private:
        Real X(Real f) const;

        const Real f0_, alpha_, beta_, delta_, x0_;
    };


    /*! The underlying \f$ f \f$ is a driftless forward; the option value is
        discounted with the curve. See AnalyticDriftedCEVEngine for an asset
        with drift.
    */
    class AnalyticCEVEngine : public VanillaOption::engine {
      public:
        AnalyticCEVEngine(Real f0, Real alpha, Real beta, Handle<YieldTermStructure> discountCurve);

        void calculate() const override;

      private:
        const ext::shared_ptr<CEVCalculator> calculator_;
        const Handle<YieldTermStructure> discountCurve_;
    };

    namespace detail {

        /*! Width of the neighbourhood of \f$ \beta = 1 \f$ inside which the
            drifted engines price with the Black-Scholes limit at the local
            lognormal volatility \f$ \alpha f_0^{\beta-1} \f$. The closed form
            degenerates there (infinite degrees of freedom).
        */
        constexpr double cevBlackLimitBandWidth = 1e-4;

        /*! Effective time of the drift-removed CEV process.

            With \f$ \tilde f = f \cdot disc(0,t) \f$ the process driven by a
            deterministic rate curve is driftless with volatility
            \f$ \alpha\, disc(0,t)^{1-\beta} \f$; a deterministic time change
            maps it to the driftless CEV process with unit volatility at

            \f[
                t_{\mathrm{eff}} = \alpha^2 \int_{start}^{end}
                    \left(\frac{disc(0,s)}{disc(0,start)}\right)^{2(1-\beta)} ds.
            \f]

            An option on \f$ f \f$ maturing at \f$ end \f$ is the undiscounted
            value of that process started at the current \f$ f \f$, at the
            strike \f$ K\,disc(0,end)/disc(0,start) \f$. The integral uses a
            64 point Gauss-Legendre rule per year of the interval.
        */
        Time cevEffectiveTime(Real alpha, Real beta,
                              const YieldTermStructure& discountCurve,
                              Time start, Time end);

    }


    //! Analytic engine for European options on a CEV asset with drift
    /*! The underlying \f$ f \f$ is an asset with drift \f$ r_t f_t \f$
        given by the discount curve, not a driftless forward as in
        AnalyticCEVEngine, which prices a forward and discounts it; the two
        engines coincide at zero rates. Inside a small neighbourhood of
        \f$ \beta = 1 \f$ the Black-Scholes limit is used; just outside it
        the closed form can still fail for short maturities or low
        volatilities, and boost then raises its own exception, not a
        QuantLib::Error. Only European exercise and plain vanilla payoffs
        are supported.
    */
    class AnalyticDriftedCEVEngine : public VanillaOption::engine {
      public:
        AnalyticDriftedCEVEngine(Real f0,
                                 Real alpha,
                                 Real beta,
                                 Handle<YieldTermStructure> discountCurve);

        void calculate() const override;

      private:
        const ext::shared_ptr<CEVCalculator> calculator_;
        const Handle<YieldTermStructure> discountCurve_;
    };
}

#endif
