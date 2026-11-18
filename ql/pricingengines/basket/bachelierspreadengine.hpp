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

/*! \file bachelierspreadengine.hpp
    \brief exact spread option pricing when both forwards are normal
*/

#ifndef quantlib_bachelier_spread_engine_hpp
#define quantlib_bachelier_spread_engine_hpp

#include <ql/instruments/basketoption.hpp>
#include <ql/processes/blackscholesprocess.hpp>
#include <ql/quote.hpp>

namespace QuantLib {

    //! Exact pricing engine for European spread options in the Bachelier model
    /*! The two forwards are jointly normal with absolute volatilities
        \f$ \sigma_1, \sigma_2 \f$ and correlation \f$ \rho \f$, so the
        spread \f$ F_1 - F_2 \f$ is normal with variance
        \f$ (\sigma_1^2 + \sigma_2^2 - 2\rho\sigma_1\sigma_2)\,T \f$ and the
        option value is the Bachelier formula applied to that spread. The
        strike may be zero or negative. Unlike the lognormal spread engines,
        no approximation is involved.

        The processes supply the spot, risk-free and dividend curves only;
        their Black volatility surfaces are not used. The absolute
        volatilities are flat and passed separately.

        \ingroup basketengines

        \test the correctness of the returned value is tested by
              put-call parity, the perfectly correlated degenerate cases and
              a Monte Carlo simulation of the two normal forwards.
    */
    class BachelierSpreadEngine : public BasketOption::engine {
      public:
        BachelierSpreadEngine(ext::shared_ptr<GeneralizedBlackScholesProcess> process1,
                              ext::shared_ptr<GeneralizedBlackScholesProcess> process2,
                              Handle<Quote> absoluteVolatility1,
                              Handle<Quote> absoluteVolatility2,
                              Real correlation);

        void calculate() const override;

      private:
        ext::shared_ptr<GeneralizedBlackScholesProcess> process1_, process2_;
        Handle<Quote> vol1_, vol2_;
        Real rho_;
    };
}

#endif
