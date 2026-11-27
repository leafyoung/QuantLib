/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
 Copyright (C) 2004 Ferdinando Ametrano
 Copyright (C) 2007 StatPro Italia srl

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

/*! \file jumpdiffusionengine.hpp
    \brief Jump diffusion (Merton 1976) engine
*/

#ifndef quantlib_jumpdiffusionengine_h
#define quantlib_jumpdiffusionengine_h

#include <ql/instruments/vanillaoption.hpp>
#include <ql/processes/merton76process.hpp>

namespace QuantLib {

    //! Jump-diffusion engine for vanilla options
    /*! \ingroup vanillaengines

        \test
        - the correctness of the returned value is tested by
          reproducing results available in literature.
        - the correctness of the returned greeks is tested by
          reproducing numerical derivatives.

        The option value is a Poisson mixture of Black-Scholes values, summed
        until the last addendum is below the relative accuracy.

        At least floor(max(lambda', lambda) T) + 1 addenda are summed, with
        lambda' the intensity times the mean jump factor and lambda the
        intensity, so that the running sum has covered the bulk of the call
        and put weights before the stopping test applies. The stopping test
        compares the last addendum with the running sum; for strikes so far
        out of the money that the sum is negligible over that range it can
        stop early, so the relative accuracy is not a bound on the error there.
    */
    class JumpDiffusionEngine : public VanillaOption::engine {
      public:
        /*! \param maxIterations the maximum number of addenda; reaching it
                   before the stopping test is met raises an error. If left
                   at Null<Size>(), it is the larger of 100 and a number of
                   terms, with a safety margin, beyond which the Poisson tail
                   mass is below the relative accuracy.
        */
        explicit JumpDiffusionEngine(ext::shared_ptr<Merton76Process>,
                                     Real relativeAccuracy_ = 1e-4,
                                     Size maxIterations = Null<Size>());
        void calculate() const override;

      private:
        ext::shared_ptr<Merton76Process> process_;
        Real relativeAccuracy_;
        Size maxIterations_;
    };

}





#endif
