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

/*! \file jarrowyildirim.hpp
    \brief Jarrow-Yildirim inflation model with Hull-White nominal and real rates
*/

#ifndef quantlib_jarrow_yildirim_hpp
#define quantlib_jarrow_yildirim_hpp

#include <ql/handle.hpp>
#include <ql/option.hpp>
#include <ql/patterns/observable.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <ql/types.hpp>

namespace QuantLib {

    //! Jarrow-Yildirim model for the nominal and real rates and the CPI
    /*! Under the nominal risk-neutral measure
        \f[
            \begin{array}{rcl}
            dr_n & = & (\theta_n(t) - a_n r_n)\,dt + \sigma_n dW_n \\
            dr_r & = & (\theta_r(t) - \rho_{rI}\sigma_r\sigma_I - a_r r_r)\,dt
                       + \sigma_r dW_r \\
            dI/I & = & (r_n - r_r)\,dt + \sigma_I dW_I
            \end{array}
        \f]
        with constant parameters, \f$ d\langle W_n,W_r\rangle = \rho_{nr}dt \f$,
        \f$ d\langle W_n,W_I\rangle = \rho_{nI}dt \f$ and
        \f$ d\langle W_r,W_I\rangle = \rho_{rI}dt \f$. The nominal and real
        Hull-White rates are fitted to the nominal and real zero curves, the
        latter being the curve of real zero-coupon bond prices
        \f$ P_r(0,T) \f$; both mean reversions must be positive.

        The index forward \f$ F(t,T) = I_t P_r(t,T)/P_n(t,T) \f$ is a
        martingale under the nominal \f$ T \f$-forward measure, so \f$ I_T \f$
        is lognormal there with total variance
        \f[
            V(0,T) = \int_0^T \sigma_I^2 + \sigma_r^2 B_r^2 + \sigma_n^2 B_n^2
              - 2\rho_{rI}\sigma_I\sigma_r B_r + 2\rho_{nI}\sigma_I\sigma_n B_n
              - 2\rho_{nr}\sigma_n\sigma_r B_n B_r \; dt,
        \f]
        where \f$ B_x = B_x(t,T) = (1-e^{-a_x(T-t)})/a_x \f$.

        The year-on-year forward paid at \f$ T \f$ is
        \f[
            E^T\left[\frac{I_T}{I_S}\right] =
            \frac{F(0,T)}{F(0,S)}\exp\left(
              \sigma_r B_r(S,T)\int_0^S e^{-a_r(S-u)}
              \left[\rho_{rI}\sigma_I + \rho_{nr}\sigma_n B_n(u,S)
                    - \sigma_r B_r(u,S)\right] du \right),
        \f]
        which follows from \f$ E^T[I_T/I_S] = E^T[P_r(S,T)/P_n(S,T)] \f$.

        Times are year fractions.

        \test the closed forms are checked against numerical quadrature and
              an exact Monte Carlo simulation of the Gaussian factors
    */
    class JarrowYildirim : public Observer, public Observable {
      public:
        JarrowYildirim(Handle<YieldTermStructure> nominalCurve,
                       Handle<YieldTermStructure> realCurve,
                       Real nominalMeanReversion,
                       Real nominalVolatility,
                       Real realMeanReversion,
                       Real realVolatility,
                       Real indexVolatility,
                       Real rhoNominalReal,
                       Real rhoNominalIndex,
                       Real rhoRealIndex,
                       Real baseIndex);

        Real nominalMeanReversion() const { return an_; }
        Real nominalVolatility() const { return sn_; }
        Real realMeanReversion() const { return ar_; }
        Real realVolatility() const { return sr_; }
        Real indexVolatility() const { return sigmaI_; }
        Real rhoNominalReal() const { return rhoNR_; }
        Real rhoNominalIndex() const { return rhoNI_; }
        Real rhoRealIndex() const { return rhoRI_; }
        Real baseIndex() const { return baseIndex_; }

        //! \f$ F(0,T) = I_0 P_r(0,T)/P_n(0,T) \f$
        Real forwardIndex(Time T) const;

        //! total variance \f$ V(0,T) \f$ of \f$ \ln I_T \f$ under the \f$ T \f$-forward measure
        Real zcInflationVariance(Time T) const;

        /*! Price of a zero-coupon inflation cap/floor paying
            \f$ \max(\pm(I_T/I_0 - K),0) \f$ at \f$ T \f$.
        */
        Real zcInflationOption(Option::Type type, Real strike, Time T) const;

        //! \f$ E^T[I_T/I_S]/(F(0,T)/F(0,S)) \f$, the convexity factor
        Real yoyConvexity(Time S, Time T) const;

        //! \f$ E^T[I_T/I_S] \f$ for \f$ 0 \le S < T \f$
        Real yoyForward(Time S, Time T) const;

        void update() override { notifyObservers(); }

      private:
        Handle<YieldTermStructure> nominalCurve_, realCurve_;
        Real an_, sn_, ar_, sr_, sigmaI_, rhoNR_, rhoNI_, rhoRI_, baseIndex_;
    };

}

#endif
