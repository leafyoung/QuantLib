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
        Hull-White rates are fitted to the nominal and real zero curves; both
        mean reversions must be positive.

        The real curve is an input, not derived by the model: a plain
        YieldTermStructure whose discount factors are the real zero-coupon
        bond prices \f$ P_r(0,T) \f$. They are tied to the index forward by
        \f[
            P_r(0,T) = P_n(0,T)\,\frac{E^T[I_T]}{I_0}.
        \f]
        For instance, from a zero-inflation curve with annually compounded
        zero rates \f$ z(T) \f$ and \f$ E^T[I_T]/I_0 = (1+z(T))^T \f$, the real
        curve has discount factors \f$ P_n(0,T)(1+z(T))^T \f$; observation lags,
        interpolation and seasonality are the user's responsibility when
        building it.

        \f$ I_T \f$ is the index level at time \f$ T \f$ itself, with no
        observation lag or interpolation: the prices below are not prices of
        CPI cap/floors on lagged fixings.

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

        Times are year fractions from the common reference date of the nominal and
        real curves, measured with their common day counter (both are enforced to
        match).

        \test the closed forms are checked against numerical quadrature and
              an exact Monte Carlo simulation of the Gaussian factors
    */
    class JarrowYildirim : public Observer, public Observable {
      public:
        /*! \param nominalTermStructure  curve of nominal discount factors; it must not
                be empty, checked at construction and at each call
            \param realTermStructure  curve of real discount factors; it must not be empty
                and must have the same reference date and day counter as the nominal
                curve, checked at construction and at each call
            \param nominalMeanReversion  \f$ a_n > 0 \f$
            \param nominalVolatility  absolute (normal) volatility \f$ \sigma_n \ge 0 \f$ of
                the nominal short rate
            \param realMeanReversion  \f$ a_r > 0 \f$
            \param realVolatility  absolute (normal) volatility \f$ \sigma_r \ge 0 \f$ of
                the real short rate
            \param indexVolatility  lognormal volatility \f$ \sigma_I \ge 0 \f$ of the index
            \param rhoNominalReal  correlation \f$ \rho_{nr} \f$
            \param rhoNominalIndex  correlation \f$ \rho_{nI} \f$
            \param rhoRealIndex  correlation \f$ \rho_{rI} \f$

            Each correlation must lie in \f$ [-1,1] \f$ and the correlation
            matrix must be positive semi-definite (determinant at least
            \f$ -10^{-12} \f$).

            The model registers with both curve handles and notifies its own
            observers when either changes; results are computed on demand from
            the current curves.
        */
        JarrowYildirim(Handle<YieldTermStructure> nominalTermStructure,
                       Handle<YieldTermStructure> realTermStructure,
                       Real nominalMeanReversion,
                       Real nominalVolatility,
                       Real realMeanReversion,
                       Real realVolatility,
                       Real indexVolatility,
                       Real rhoNominalReal,
                       Real rhoNominalIndex,
                       Real rhoRealIndex);

        const Handle<YieldTermStructure>& nominalTermStructure() const {
            return nominalTermStructure_;
        }
        const Handle<YieldTermStructure>& realTermStructure() const { return realTermStructure_; }
        Real nominalMeanReversion() const { return aN_; }
        Real nominalVolatility() const { return sigmaN_; }
        Real realMeanReversion() const { return aR_; }
        Real realVolatility() const { return sigmaR_; }
        Real indexVolatility() const { return sigmaI_; }
        Real rhoNominalReal() const { return rhoNR_; }
        Real rhoNominalIndex() const { return rhoNI_; }
        Real rhoRealIndex() const { return rhoRI_; }

        //! forward index over the current index level, \f$ F(0,T)/I_0 = P_r(0,T)/P_n(0,T) \f$
        /*! Defined for \f$ T \ge 0 \f$. */
        Real forwardIndexRatio(Time T) const;

        //! total variance of \f$ \ln I_T \f$ under the nominal \f$ T \f$-forward measure
        /*! This is \f$ V(0,T) \f$, defined for \f$ T \ge 0 \f$ and never negative.
         */
        Real zeroInflationVariance(Time T) const;

        //! price of a zero-coupon inflation cap/floor
        /*! Per unit notional and discounted on the nominal curve.
            Option::Call is the zero-coupon cap, paying \f$ \max(I_T/I_0 - K,0) \f$, and
            Option::Put is the floor, paying \f$ \max(K - I_T/I_0,0) \f$. Defined for
            \f$ T > 0 \f$.

            The strike \f$ K \ge 0 \f$ is a ratio on \f$ I_T/I_0 \f$, not a rate or an index
            level; for a market zero-coupon cap/floor quoted with an annual strike rate
            \f$ k \f$ it is \f$ K = (1+k)^T \f$.
        */
        Real zeroInflationOption(Option::Type type, Real strike, Time T) const;

        //! year-on-year convexity factor \f$ E^T[I_T/I_S]/(F(0,T)/F(0,S)) \f$
        /*! Defined for \f$ 0 \le S < T \f$. */
        Real yoyConvexity(Time S, Time T) const;

        //! year-on-year forward ratio \f$ E^T[I_T/I_S] \f$
        /*! Defined for \f$ 0 \le S < T \f$. It is one plus the year-on-year
            forward return over \f$ [S,T] \f$ (\f$ 1 + (T-S)f \f$ for an
            annualized forward rate \f$ f \f$), not a rate itself.
        */
        Real yoyForwardRatio(Time S, Time T) const;

        void update() override { notifyObservers(); }

      private:
        void checkCurves() const;

        Handle<YieldTermStructure> nominalTermStructure_, realTermStructure_;
        Real aN_, sigmaN_, aR_, sigmaR_, sigmaI_, rhoNR_, rhoNI_, rhoRI_;
    };

}

#endif
