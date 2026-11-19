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

/*! \file backwardlookingvariancetime.hpp
    \brief variance time of a backward-looking (compounded overnight) rate
*/

#ifndef quantlib_backward_looking_variance_time_hpp
#define quantlib_backward_looking_variance_time_hpp

#include <ql/types.hpp>

namespace QuantLib {

    //! Total-variance time of a backward-looking rate observed from time zero
    /*! A compounded overnight rate \f$ R(t) \f$ over the accrual period
        \f$ [\tau_0, \tau_1] \f$ is a martingale under the
        \f$ \tau_1 \f$-forward measure. Inside the accrual period its
        volatility is damped by
        \f[
            \psi(t) = \min\left(1, \left(\frac{\tau_1 - t}{\tau_1 - \tau_0}
                       \right)^q\right), \qquad q > 0,
        \f]
        which accounts for the fixings already realised. For a constant
        volatility \f$ \sigma \f$ the total variance of \f$ R(\tau_1) \f$
        (of its logarithm, possibly shifted, for lognormal volatilities) is
        \f$ \sigma^2 \Theta \f$ with
        \f[
            \Theta = \int_0^{\tau_1} \psi(t)^2 \, dt
                    = s + \frac{(\tau_1 - s)\, x^{2q}}{2q + 1},
            \qquad s = \max(\tau_0, 0), \quad
            x = \frac{\tau_1 - s}{\tau_1 - \tau_0}.
        \f]
        The exponent \f$ q = 1 \f$ is the linear damping from one at the
        start of the accrual period to zero at its end.

        If the accrual period has no part in the future, that is
        \f$ \tau_1 \le s \f$ (in particular a single fixing, or a period
        that has ended), nothing is damped and the result is \f$ s \f$.

        The power family of damping profiles follows Willems (arXiv:2004.04501,
        Section 3).

        \param fixingStart  time of the first fixing of the accrual period
        \param fixingEnd    time of the last fixing; must not precede
                            \p fixingStart
        \param dampingExponent  the exponent \f$ q \f$; must be positive
    */
    Time backwardLookingVarianceTime(Time fixingStart, Time fixingEnd, Real dampingExponent = 1.0);

}

#endif
