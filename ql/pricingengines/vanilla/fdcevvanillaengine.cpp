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

#include <ql/exercise.hpp>
#include <ql/methods/finitedifferences/meshers/concentrating1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmcev1dmesher.hpp>
#include <ql/methods/finitedifferences/meshers/fdmmeshercomposite.hpp>
#include <ql/methods/finitedifferences/operators/fdmcevop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearoplayout.hpp>
#include <ql/methods/finitedifferences/solvers/fdm1dimsolver.hpp>
#include <ql/methods/finitedifferences/stepconditions/fdmstepconditioncomposite.hpp>
#include <ql/methods/finitedifferences/utilities/fdmdiscountdirichletboundary.hpp>
#include <ql/methods/finitedifferences/utilities/fdminnervaluecalculator.hpp>
#include <ql/methods/finitedifferences/utilities/fdmtimedepdirichletboundary.hpp>
#include <ql/pricingengines/blackformula.hpp>
#include <ql/pricingengines/vanilla/analyticcevengine.hpp>
#include <ql/pricingengines/vanilla/fdcevvanillaengine.hpp>
#include <ql/termstructures/yieldtermstructure.hpp>
#include <algorithm>
#include <utility>

namespace QuantLib {

    namespace {
        class PriceAtBoundary {
          public:
            PriceAtBoundary(Time maturityTime,
                            ext::shared_ptr<StrikedTypePayoff> payoff,
                            ext::shared_ptr<YieldTermStructure> rTS,
                            Real boundary, Real alpha, Real beta)
            : maturityTime_(maturityTime), payoff_(std::move(payoff)),
              rTS_(std::move(rTS)),
              calculator_(boundary, 1.0, beta),
              boundary_(boundary), alpha_(alpha), beta_(beta) {}

            Real operator()(Real rollbackTime) const {
                // the rollback destination is strictly below the maturity,
                // so the evaluation window has positive length and stays
                // within the curve's range. A window shorter than about a
                // day is replaced by one of that length: the noncentral
                // chi-squared evaluation fails for very short windows,
                // and the boundary value hardly changes over them.
                const Time t = std::min<Real>(
                    rollbackTime,
                    maturityTime_ - std::min<Real>(1.0/365.0, 0.5*maturityTime_));
                const DiscountFactor df =
                    rTS_->discount(maturityTime_) / rTS_->discount(t);

                if (std::fabs(beta_ - 1.0) < cevBlackLimitBandWidth) {
                    // the CEV price converges to its Black-Scholes limit as
                    // beta approaches one; see AnalyticCEVEngine
                    return blackFormula(
                        payoff_->optionType(), payoff_->strike(),
                        boundary_/df,
                        alpha_*std::sqrt(maturityTime_ - t), df);
                }

                // drift-removed unit volatility process, see cevEffectiveTime
                try {
                    return calculator_.value(
                        payoff_->optionType(),
                        payoff_->strike()*df,
                        cevEffectiveTime(alpha_, beta_, rTS_, t, maturityTime_));
                } catch (const std::exception&) {
                    // the noncentral chi-squared evaluation fails for beta
                    // close to one far from the money; the value there is
                    // that of the local Black-Scholes volatility at the
                    // boundary, which beta close to one makes accurate
                    return blackFormula(
                        payoff_->optionType(), payoff_->strike(),
                        boundary_/df,
                        alpha_*std::pow(boundary_, beta_ - 1.0) *
                            std::sqrt(maturityTime_ - t),
                        df);
                }
            }

          private:
            const Time maturityTime_;
            const ext::shared_ptr<StrikedTypePayoff> payoff_;
            const ext::shared_ptr<YieldTermStructure> rTS_;
            const CEVCalculator calculator_;
            const Real boundary_, alpha_, beta_;
        };
    }

    FdCEVVanillaEngine::FdCEVVanillaEngine(Real f0,
                                           Real alpha,
                                           Real beta,
                                           Handle<YieldTermStructure> discountCurve,
                                           Size tGrid,
                                           Size xGrid,
                                           Size dampingSteps,
                                           Real scalingFactor,
                                           Real eps,
                                           const FdmSchemeDesc& schemeDesc)
    : f0_(f0), alpha_(alpha), beta_(beta), discountCurve_(std::move(discountCurve)), tGrid_(tGrid),
      xGrid_(xGrid), dampingSteps_(dampingSteps), scalingFactor_(scalingFactor), eps_(eps),
      schemeDesc_(schemeDesc) {
        registerWith(discountCurve_);
    }

    void FdCEVVanillaEngine::calculate() const {

        // 1. Mesher
        const ext::shared_ptr<StrikedTypePayoff> payoff =
            ext::dynamic_pointer_cast<StrikedTypePayoff>(arguments_.payoff);
        QL_REQUIRE(payoff, "non-striked payoff given");

        const ext::shared_ptr<YieldTermStructure> rTS =
            discountCurve_.currentLink();

        const DayCounter dc = rTS->dayCounter();

        const Date referenceDate = rTS->referenceDate();
        const Date maturityDate = arguments_.exercise->lastDate();
        const Time maturityTime = dc.yearFraction(referenceDate, maturityDate);

        QL_REQUIRE(maturityTime > 0.0,
                   "the option matures at or before the discount curve's "
                   "reference date");

        const ext::shared_ptr<Fdm1dMesher> cevMesher =
            ext::make_shared<FdmCEV1dMesher>(
                xGrid_,
                f0_, alpha_, beta_,
                maturityTime, eps_, scalingFactor_,
                std::make_pair(payoff->strike(), 0.1));

        const Real lowerBound = cevMesher->locations().front();
        const Real upperBound = cevMesher->locations().back();

        const ext::shared_ptr<FdmMesher> mesher =
           ext::make_shared<FdmMesherComposite>(cevMesher);

        // 2. Calculator
        const ext::shared_ptr<FdmInnerValueCalculator> calculator =
            ext::make_shared<FdmCellAveragingInnerValue>(payoff, mesher, 0);

        // 3. Step conditions
        const ext::shared_ptr<FdmStepConditionComposite> conditions =
            FdmStepConditionComposite::vanillaComposite(
                DividendSchedule(), arguments_.exercise,
                mesher, calculator,
                referenceDate, dc);

        // 4. Boundary conditions
        FdmBoundaryConditionSet boundaries;

        const PriceAtBoundary upperBoundPrice(
            maturityTime, payoff, rTS, upperBound, alpha_, beta_);

        boundaries.push_back(ext::make_shared<FdmTimeDepDirichletBoundary>(
            mesher, std::function<Real (Real)>(upperBoundPrice),
            0, FdmTimeDepDirichletBoundary::Upper));

        if (lowerBound > 0.0) {
            // the asset has drift, so the value at the lower bound is the
            // drifted CEV price there, as at the upper bound
            const PriceAtBoundary lowerBoundPrice(
                maturityTime, payoff, rTS, lowerBound, alpha_, beta_);

            boundaries.push_back(
                ext::make_shared<FdmTimeDepDirichletBoundary>(
                    mesher, std::function<Real (Real)>(lowerBoundPrice),
                    0, FdmTimeDepDirichletBoundary::Lower));
        }
        else if (beta_ < 1.0) {
            // the mesh reaches zero, which is absorbing for beta < 1
            const Real terminalCashFlow = (*payoff)(lowerBound);

            boundaries.push_back(
                ext::make_shared<FdmDiscountDirichletBoundary>(
                    mesher, rTS, maturityTime, terminalCashFlow,
                    0, FdmTimeDepDirichletBoundary::Lower));
        }

        // 5. Solver
        const FdmSolverDesc solverDesc = {
            mesher, boundaries, conditions,
            calculator, maturityTime, tGrid_, dampingSteps_
        };

        const ext::shared_ptr<FdmLinearOpComposite> op =
            ext::make_shared<FdmCEVOp>(
               mesher, discountCurve_.currentLink(), f0_, alpha_, beta_, 0);

        const auto solver =
            ext::make_shared<Fdm1DimSolver>(solverDesc, schemeDesc_, op);

        results_.value = solver->interpolateAt(f0_);
        results_.delta = solver->derivativeX(f0_);
        results_.gamma = solver->derivativeXX(f0_);
        results_.theta = solver->thetaAt(f0_);
    }
}
