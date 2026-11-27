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

    //! Finite-difference engine for options on a CEV asset
    /*! As in AnalyticCEVEngine, the underlying is an asset with drift
        given by the discount curve, not a driftless forward. FdSabrVanillaEngine
        models a driftless forward, so the two agree only for a zero rate.
        The mesher requires \f$ \beta \neq 1 \f$ (and fails to converge
        closer to one than about 3e-4 on a wide mesh), unlike
        AnalyticCEVEngine, which accepts \f$ \beta = 1 \f$.
        The boundary values are the European prices at the mesh bounds, which
        is exact for European exercise; with early exercise they are an
        approximation that improves as the mesh widens (smaller eps).
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
}

#endif
