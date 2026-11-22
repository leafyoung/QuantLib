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

#include <ql/cashflows/backwardlookingvariancetime.hpp>
#include <ql/errors.hpp>
#include <algorithm>
#include <cmath>

namespace QuantLib {

    Time backwardLookingVarianceTime(Time fixingStart, Time fixingEnd, Real dampingExponent) {
        QL_REQUIRE(fixingEnd >= fixingStart,
                   "fixing end time (" << fixingEnd << ") must not precede fixing start time ("
                                       << fixingStart << ")");
        QL_REQUIRE(dampingExponent > 0.0,
                   "damping exponent (" << dampingExponent << ") must be positive");
        Time elapsed = std::max<Real>(fixingStart, 0.0);
        // no part of the accrual period lies in the future: nothing is damped
        if (fixingEnd <= elapsed)
            return elapsed;
        Real remaining = fixingEnd - elapsed;
        Real ratio = remaining / (fixingEnd - fixingStart);
        return elapsed +
               remaining * std::pow(ratio, 2.0 * dampingExponent) / (2.0 * dampingExponent + 1.0);
    }

}
