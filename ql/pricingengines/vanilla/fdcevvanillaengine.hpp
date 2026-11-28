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

/*! \file fdcevvanillaengine.hpp
    \brief Finite-Differences pricing engine for the CEV model
*/

#ifndef quantlib_fd_cev_vanilla_engine_hpp
#define quantlib_fd_cev_vanilla_engine_hpp

#include <ql/handle.hpp>
#include <ql/instruments/vanillaoption.hpp>
#include <ql/methods/finitedifferences/solvers/fdmbackwardsolver.hpp>

namespace QuantLib {
    class YieldTermStructure;

    /*! The underlying \f$ f \f$ is a driftless forward; the option value is
        discounted with the curve. See FdDriftedCEVVanillaEngine for an asset
        with drift.
    */
    class FdCEVVanillaEngine : public VanillaOption::engine {
      public:
        FdCEVVanillaEngine(Real f0,
                           Real alpha,
                           Real beta,
                           Handle<YieldTermStructure> discountCurve,
                           Size tGrid = 50,
                           Size xGrid = 400,
                           Size dampingSteps = 0,
                           Real scalingFactor = 1.0,
                           Real eps = 1e-4,
                           const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas());

        void calculate() const override;

      private:
        const Real f0_, alpha_, beta_;
        const Handle<YieldTermStructure> discountCurve_;
        const Size tGrid_, xGrid_, dampingSteps_;
        const Real scalingFactor_, eps_;
        const FdmSchemeDesc schemeDesc_;
    };

    //! Finite-difference engine for options on a CEV asset with drift
    /*! The underlying is an asset with drift given by the discount curve,
        as in AnalyticDriftedCEVEngine, not a driftless forward as in
        FdCEVVanillaEngine and FdSabrVanillaEngine; those agree with this
        engine only at zero rates.

        The mesher requires \f$ \beta \neq 1 \f$, and its noncentral
        chi-squared evaluation can fail, mostly for \f$ |\beta - 1| \f$ up
        to about 1e-3 but also sporadically elsewhere.

        The boundary values are European prices at the mesh bounds, so early
        exercise is approximate. For \f$ \beta < 1 \f$ it improves as the
        mesh widens (smaller eps); for \f$ \beta > 1 \f$ it does not, and an
        eps below the default can give wrong prices.

        Only plain vanilla payoffs are supported. The kinked payoff needs
        \p dampingSteps of 2 for accurate prices and Greeks; the default of
        0 can give oscillating Greeks and slightly negative prices far out
        of the money. \p tGrid has to grow with the drift (the rate times
        the maturity of about 2 or more at a volatility of 1% or less) and
        with the maturity for \f$ \beta > 1 \f$ calls, where 50 steps can be
        off by up to about 20%.
    */
    class FdDriftedCEVVanillaEngine : public VanillaOption::engine {
      public:
        FdDriftedCEVVanillaEngine(Real f0,
                                  Real alpha,
                                  Real beta,
                                  Handle<YieldTermStructure> discountCurve,
                                  Size tGrid = 50,
                                  Size xGrid = 400,
                                  Size dampingSteps = 0,
                                  Real scalingFactor = 1.0,
                                  Real eps = 1e-4,
                                  const FdmSchemeDesc& schemeDesc = FdmSchemeDesc::Douglas());

        void calculate() const override;

      private:
        const Real f0_, alpha_, beta_;
        const Handle<YieldTermStructure> discountCurve_;
        const Size tGrid_, xGrid_, dampingSteps_;
        const Real scalingFactor_, eps_;
        const FdmSchemeDesc schemeDesc_;
    };
}

#endif
