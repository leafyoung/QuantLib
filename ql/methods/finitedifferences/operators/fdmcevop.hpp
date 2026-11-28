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

/*! \file fdmcevop.hpp
    \brief FDM operator for the Constant Elasticity of Variance (CEV) model
*/

#ifndef quantlib_fdm_cev_op_hpp
#define quantlib_fdm_cev_op_hpp

#include <ql/methods/finitedifferences/operators/ninepointlinearop.hpp>
#include <ql/methods/finitedifferences/operators/triplebandlinearop.hpp>
#include <ql/methods/finitedifferences/operators/fdmlinearopcomposite.hpp>

namespace QuantLib {

    //! constant elasticity of variance process (absorbing boundary at f=0)
    /*! \f[
         df_t = \alpha f_t^\beta \mathrm{d}W_t
        \f]
    */

    class FdmMesher;
    class YieldTermStructure;

    class FdmCEVOp : public FdmLinearOpComposite {
      public:

        FdmCEVOp(
            const ext::shared_ptr<FdmMesher>& mesher,
            const ext::shared_ptr<YieldTermStructure>& rTS,
            Real f0, Real alpha, Real beta,
            Size direction);

        Size size() const override;
        void setTime(Time t1, Time t2) override;

        Array apply(const Array& r) const override;
        Array apply_mixed(const Array& r) const override;
        Array apply_direction(Size direction, const Array& r) const override;
        Array solve_splitting(Size direction, const Array& r, Real s) const override;
        Array preconditioner(const Array& r, Real s) const override;

        std::vector<SparseMatrix> toMatrixDecomp() const override;

      private:
        const ext::shared_ptr<YieldTermStructure> rTS_;
        const Size direction_;
        const TripleBandLinearOp dxxMap_;
        TripleBandLinearOp mapT_;
    };

    //! constant elasticity of variance process for an asset with drift
    /*! \f[
         df_t = r_t f_t \mathrm{d}t + \alpha f_t^\beta \mathrm{d}W_t
        \f]

        For \f$ \beta < 1 \f$ the process has an absorbing boundary at
        \f$ f = 0 \f$; for \f$ \beta > 1 \f$ zero is unattainable and
        the discounted process is a strict local martingale.

        The first-derivative (drift) rows at the mesh end points that carry
        Dirichlet conditions are emptied, so the operator is meant to be
        used with Dirichlet boundary conditions there: the mesh's lower end
        when \f$ \beta < 1 \f$ or it lies above zero, and its upper end.
    */

    class FdmDriftedCEVOp : public FdmLinearOpComposite {
      public:

        FdmDriftedCEVOp(
            const ext::shared_ptr<FdmMesher>& mesher,
            const ext::shared_ptr<YieldTermStructure>& rTS,
            Real f0, Real alpha, Real beta,
            Size direction);

        Size size() const override;
        void setTime(Time t1, Time t2) override;

        Array apply(const Array& r) const override;
        Array apply_mixed(const Array& r) const override;
        Array apply_direction(Size direction, const Array& r) const override;
        Array solve_splitting(Size direction, const Array& r, Real s) const override;
        Array preconditioner(const Array& r, Real s) const override;

        std::vector<SparseMatrix> toMatrixDecomp() const override;

      private:
        const ext::shared_ptr<YieldTermStructure> rTS_;
        const Size direction_;
        const TripleBandLinearOp dxMap_;
        const TripleBandLinearOp dxxMap_;
        TripleBandLinearOp mapT_;
    };
}

#endif
