//! The vendor BLAS/LAPACK arm: CBLAS `?gemm` plus LAPACKE drivers.
//!
//! The harness declares its own `extern "C"` bindings rather than adding a
//! binding crate: `blas-src`/`lapack-src` (pulled by the `link-*` features in
//! `Cargo.toml`) only emit the native link flags, and the campaign pins the
//! symbol set it calls. Every call goes through LAPACKE's row-major/col-major
//! layout argument, so the `order` axis is a native route, not a transpose the
//! harness performs.
//!
//! # No allocation and no layout conversion inside the clock
//!
//! The measured call must not allocate (docs/reliable-measurement.md,
//! "Allocation and the clock"): every driver is called through its `_work` entry
//! point with a workspace this module owns, sized once at op construction and
//! stored in the op's state. The non-`_work` wrappers allocate that workspace per
//! call (`lapacke_dgesdd.c`, `lapacke_dsyevd.c`, `lapacke_dgeqrf.c`), which is
//! exactly the tax the comparison must not contain.
//!
//! The LAPACK drivers also run **column-major only**, because LAPACKE's row-major
//! path allocates a transposed copy and transposes in and out of it *inside* the
//! call — in every `_work` variant (reference `lapacke_dgeqrf_work.c:70-92`), so
//! no argument removes it. A C-order cell of a driver family therefore has no
//! route that leaves the clock allocation- and conversion-free, and it is
//! reported as `skipped` with that reason instead of as a number: the faer rows
//! factor a row-major input where it is, so a C-order vendor number would charge
//! a copy to one side only. `gemm` keeps both orders: CBLAS's row-major entry
//! point reorders its operands and moves no data.
//!
//! The complex ABI is the one flat 16-byte `{re, im}` struct MKL documents as
//! `lapack_complex_double`/`MKL_Complex16`; `num_complex::Complex64` is that
//! layout. (An OpenBLAS built with C99 `double complex` would differ, which is
//! why the MKL arm is the one the campaign runs.) AOCL's LAPACKE declares C99
//! `double _Complex`, whose in-memory layout is the same pair; the harness passes
//! complex values to it only by pointer, so the by-value difference never arises.
//!
//! `triangular_solve` is CBLAS `?trsm` (in place, no workspace; both orders, as
//! `gemm`). `householder_apply` is `?larfb` for the block form and `?ormqr`/`?unmqr`
//! for the sequence form, on a `?geqrf`/`?larft` fixture built outside the clock.
//! `full_piv_lu` is `?getc2`, called through its Fortran symbol because LAPACKE has
//! no wrapper for it: the **unblocked** complete-pivoting LU, square shapes only (a
//! non-square cell is `skipped`), a route difference against the faer rows' blocked
//! `lu_in_place` that is recorded in [`ROUTE_DIFFERENCES`] and printed by `fbench
//! info`. `?trsm` is bound for all four dtypes; `?getc2`, `?larfb`/`?larft` and
//! `?ormqr`/`?unmqr` for `f64`/`c64`, the dtypes the LAPACK drivers have here.
//!
//! Families with no LAPACK route the harness measures (`upgrade_factor`, `eig`)
//! return an error and are recorded as `skipped`, never as a number.

#[cfg(any(feature = "link-mkl", feature = "link-openblas", feature = "link-aocl"))]
mod imp {
    use crate::op::{op_from, Op};
    use crate::shared::*;
    use core::ffi::{c_char, c_void};
    use num_complex::Complex64;

    const COL_MAJOR: i32 = 102;
    const ROW_MAJOR: i32 = 101;
    const JOB_N: c_char = b'N' as c_char;
    const JOB_S: c_char = b'S' as c_char;
    const JOB_V: c_char = b'V' as c_char;
    const UPLO_L: c_char = b'L' as c_char;
    const SIDE_L: c_char = b'L' as c_char;
    const TRANS_N: c_char = b'N' as c_char;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_UPPER: i32 = 121;
    const CBLAS_NON_UNIT: i32 = 131;
    const CBLAS_LEFT: i32 = 141;
    const DIRECT_F: c_char = b'F' as c_char;
    const STOREV_C: c_char = b'C' as c_char;

    unsafe extern "C" {
        fn cblas_sgemm(
            layout: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
        fn cblas_cgemm(
            layout: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const c_void,
            a: *const c_void,
            lda: i32,
            b: *const c_void,
            ldb: i32,
            beta: *const c_void,
            c: *mut c_void,
            ldc: i32,
        );
        fn cblas_dgemm(
            layout: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *const f64,
            ldb: i32,
            beta: f64,
            c: *mut f64,
            ldc: i32,
        );
        fn cblas_zgemm(
            layout: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: *const c_void,
            a: *const c_void,
            lda: i32,
            b: *const c_void,
            ldb: i32,
            beta: *const c_void,
            c: *mut c_void,
            ldc: i32,
        );
        // `B <- A^{-1} B` with `A` upper triangular, on the left, in place: the `triangular_solve`
        // route. Like `?gemm`, the CBLAS row-major entry point swaps the side and the triangle
        // and moves no data, so both orders are a native route.
        fn cblas_strsm(
            layout: i32,
            side: i32,
            uplo: i32,
            transa: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *mut f32,
            ldb: i32,
        );
        fn cblas_dtrsm(
            layout: i32,
            side: i32,
            uplo: i32,
            transa: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: f64,
            a: *const f64,
            lda: i32,
            b: *mut f64,
            ldb: i32,
        );
        fn cblas_ctrsm(
            layout: i32,
            side: i32,
            uplo: i32,
            transa: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: *const c_void,
            a: *const c_void,
            lda: i32,
            b: *mut c_void,
            ldb: i32,
        );
        fn cblas_ztrsm(
            layout: i32,
            side: i32,
            uplo: i32,
            transa: i32,
            diag: i32,
            m: i32,
            n: i32,
            alpha: *const c_void,
            a: *const c_void,
            lda: i32,
            b: *mut c_void,
            ldb: i32,
        );
        // The `_work` entry points: the caller owns the workspace. Prototypes are
        // MKL's `mkl_lapacke.h` and the vendored netlib `lapacke.h`, which agree
        // on every one of them except that only the complex `gesdd` takes `rwork`.
        fn LAPACKE_dgeqrf_work(
            layout: i32,
            m: i32,
            n: i32,
            a: *mut f64,
            lda: i32,
            tau: *mut f64,
            work: *mut f64,
            lwork: i32,
        ) -> i32;
        fn LAPACKE_zgeqrf_work(
            layout: i32,
            m: i32,
            n: i32,
            a: *mut Complex64,
            lda: i32,
            tau: *mut Complex64,
            work: *mut Complex64,
            lwork: i32,
        ) -> i32;
        fn LAPACKE_dorgqr_work(
            layout: i32,
            m: i32,
            n: i32,
            k: i32,
            a: *mut f64,
            lda: i32,
            tau: *const f64,
            work: *mut f64,
            lwork: i32,
        ) -> i32;
        fn LAPACKE_zungqr_work(
            layout: i32,
            m: i32,
            n: i32,
            k: i32,
            a: *mut Complex64,
            lda: i32,
            tau: *const Complex64,
            work: *mut Complex64,
            lwork: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_dormqr_work(
            layout: i32,
            side: c_char,
            trans: c_char,
            m: i32,
            n: i32,
            k: i32,
            a: *const f64,
            lda: i32,
            tau: *const f64,
            c: *mut f64,
            ldc: i32,
            work: *mut f64,
            lwork: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_zunmqr_work(
            layout: i32,
            side: c_char,
            trans: c_char,
            m: i32,
            n: i32,
            k: i32,
            a: *const Complex64,
            lda: i32,
            tau: *const Complex64,
            c: *mut Complex64,
            ldc: i32,
            work: *mut Complex64,
            lwork: i32,
        ) -> i32;
        fn LAPACKE_dgetrf_work(
            layout: i32,
            m: i32,
            n: i32,
            a: *mut f64,
            lda: i32,
            ipiv: *mut i32,
        ) -> i32;
        fn LAPACKE_zgetrf_work(
            layout: i32,
            m: i32,
            n: i32,
            a: *mut Complex64,
            lda: i32,
            ipiv: *mut i32,
        ) -> i32;
        fn LAPACKE_dpotrf_work(layout: i32, uplo: c_char, n: i32, a: *mut f64, lda: i32) -> i32;
        fn LAPACKE_zpotrf_work(
            layout: i32,
            uplo: c_char,
            n: i32,
            a: *mut Complex64,
            lda: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_dgesdd_work(
            layout: i32,
            jobz: c_char,
            m: i32,
            n: i32,
            a: *mut f64,
            lda: i32,
            s: *mut f64,
            u: *mut f64,
            ldu: i32,
            vt: *mut f64,
            ldvt: i32,
            work: *mut f64,
            lwork: i32,
            iwork: *mut i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_zgesdd_work(
            layout: i32,
            jobz: c_char,
            m: i32,
            n: i32,
            a: *mut Complex64,
            lda: i32,
            s: *mut f64,
            u: *mut Complex64,
            ldu: i32,
            vt: *mut Complex64,
            ldvt: i32,
            work: *mut Complex64,
            lwork: i32,
            rwork: *mut f64,
            iwork: *mut i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_dsyevd_work(
            layout: i32,
            jobz: c_char,
            uplo: c_char,
            n: i32,
            a: *mut f64,
            lda: i32,
            w: *mut f64,
            work: *mut f64,
            lwork: i32,
            iwork: *mut i32,
            liwork: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_zheevd_work(
            layout: i32,
            jobz: c_char,
            uplo: c_char,
            n: i32,
            a: *mut Complex64,
            lda: i32,
            w: *mut f64,
            work: *mut Complex64,
            lwork: i32,
            rwork: *mut f64,
            lrwork: i32,
            iwork: *mut i32,
            liwork: i32,
        ) -> i32;
        // `?getc2`: LU with complete pivoting, unblocked and square only (`full_piv_lu`).
        // LAPACKE has no wrapper for it (it is an auxiliary routine; MKL, OpenBLAS and AOCL's
        // libflame all export only the Fortran symbol), so it is called through the Fortran ABI:
        // every argument by reference, LP64 integers, column-major, and `info` an out-argument.
        // It takes no character argument, so there is no hidden string-length parameter.
        fn dgetc2_(
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            ipiv: *mut i32,
            jpiv: *mut i32,
            info: *mut i32,
        );
        fn zgetc2_(
            n: *const i32,
            a: *mut Complex64,
            lda: *const i32,
            ipiv: *mut i32,
            jpiv: *mut i32,
            info: *mut i32,
        );
        // `?larft` forms the triangular factor of a block reflector (outside the clock), and
        // `?larfb` applies the block reflector: the block form of `householder_apply`.
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_dlarft_work(
            layout: i32,
            direct: c_char,
            storev: c_char,
            n: i32,
            k: i32,
            v: *const f64,
            ldv: i32,
            tau: *const f64,
            t: *mut f64,
            ldt: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_zlarft_work(
            layout: i32,
            direct: c_char,
            storev: c_char,
            n: i32,
            k: i32,
            v: *const Complex64,
            ldv: i32,
            tau: *const Complex64,
            t: *mut Complex64,
            ldt: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_dlarfb_work(
            layout: i32,
            side: c_char,
            trans: c_char,
            direct: c_char,
            storev: c_char,
            m: i32,
            n: i32,
            k: i32,
            v: *const f64,
            ldv: i32,
            t: *const f64,
            ldt: i32,
            c: *mut f64,
            ldc: i32,
            work: *mut f64,
            ldwork: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn LAPACKE_zlarfb_work(
            layout: i32,
            side: c_char,
            trans: c_char,
            direct: c_char,
            storev: c_char,
            m: i32,
            n: i32,
            k: i32,
            v: *const Complex64,
            ldv: i32,
            t: *const Complex64,
            ldt: i32,
            c: *mut Complex64,
            ldc: i32,
            work: *mut Complex64,
            ldwork: i32,
        ) -> i32;
        fn LAPACKE_dgesv_work(
            layout: i32,
            n: i32,
            nrhs: i32,
            a: *mut f64,
            lda: i32,
            ipiv: *mut i32,
            b: *mut f64,
            ldb: i32,
        ) -> i32;
        fn LAPACKE_zgesv_work(
            layout: i32,
            n: i32,
            nrhs: i32,
            a: *mut Complex64,
            lda: i32,
            ipiv: *mut i32,
            b: *mut Complex64,
            ldb: i32,
        ) -> i32;
    }

    /// The vendor calls every declared dtype has a route for: the CBLAS level-3 routines
    /// `cblas_?gemm` and `cblas_?trsm`. They are a capability of their own because they exist
    /// for all four dtypes while the LAPACK drivers below exist, in this harness, for
    /// `f64`/`c64` only: a dtype that has the former must not be made to carry dummy
    /// implementations of the latter.
    trait LapackBlas: Scalar {
        #[allow(clippy::too_many_arguments)]
        fn gemm(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &[Self],
            b: &[Self],
            c: &mut [Self],
            lda: i32,
            ldb: i32,
            ldc: i32,
        );
        /// `b <- u^{-1} b` in place, with `u` the `n x n` upper triangular (non-unit) matrix and
        /// `b` `n x nrhs`: CBLAS `?trsm` on the left, no transpose, no workspace.
        fn trsm(layout: i32, n: usize, nrhs: usize, u: &[Self], ldu: i32, b: &mut [Self], ldb: i32);
    }

    /// The vendor's LAPACK drivers, every one through its `_work` entry point.
    ///
    /// `work`/`rwork`/`iwork` are the caller's: they are sized once, outside the
    /// clock, by the `*_lwork`/`*_scratch` queries below and stored in the op.
    trait LapackScalar: LapackBlas {
        /// The optimum a `lwork = -1` query returned, as an element count.
        fn as_len(self) -> usize;

        fn geqrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [Self],
            lda: i32,
            tau: &mut [Self],
            work: &mut [Self],
        ) -> i32;
        fn orgqr(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &mut [Self],
            lda: i32,
            tau: &[Self],
            work: &mut [Self],
        ) -> i32;
        /// `C <- Q C` with the `k` Householder reflectors `geqrf` left in `a`.
        #[allow(clippy::too_many_arguments)]
        fn ormqr(
            layout: i32,
            m: usize,
            ncols: usize,
            k: usize,
            a: &[Self],
            lda: i32,
            tau: &[Self],
            c: &mut [Self],
            ldc: i32,
            work: &mut [Self],
        ) -> i32;
        fn getrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
        ) -> i32;
        fn potrf(layout: i32, n: usize, a: &mut [Self], lda: i32) -> i32;
        fn gesv(
            layout: i32,
            n: usize,
            a: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
            b: &mut [Self],
            ldb: i32,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn gesdd(
            layout: i32,
            jobz: c_char,
            m: usize,
            n: usize,
            a: &mut [Self],
            lda: i32,
            s: &mut [f64],
            u: &mut [Self],
            ldu: i32,
            vt: &mut [Self],
            ldvt: i32,
            work: &mut [Self],
            rwork: &mut [f64],
            iwork: &mut [i32],
        ) -> i32;
        fn evd(
            layout: i32,
            jobz: c_char,
            n: usize,
            a: &mut [Self],
            lda: i32,
            w: &mut [f64],
            work: &mut [Self],
            rwork: &mut [f64],
            iwork: &mut [i32],
        ) -> i32;

        /// `P A Q = L U` with complete pivoting, in place: `?getc2`. Unblocked and square
        /// only; `ipiv`/`jpiv` are LAPACK's 1-based sequential row and column interchanges.
        fn getc2(
            layout: i32,
            n: usize,
            a: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
            jpiv: &mut [i32],
        ) -> i32;
        /// The `k x k` upper triangular `T` of the block reflector `H = I - V T V^H` whose `k`
        /// columnwise, forward-ordered reflectors `?geqrf` left in `v` (`?larft`).
        #[allow(clippy::too_many_arguments)]
        fn larft(
            layout: i32,
            n: usize,
            k: usize,
            v: &[Self],
            ldv: i32,
            tau: &[Self],
            t: &mut [Self],
            ldt: i32,
        ) -> i32;
        /// `C <- H C` with `H = I - V T V^H` (`?larfb`, side L, no transpose, forward,
        /// columnwise). `work` is the caller's `ncols x k` buffer, `ldwork = ncols`.
        #[allow(clippy::too_many_arguments)]
        fn larfb(
            layout: i32,
            m: usize,
            ncols: usize,
            k: usize,
            v: &[Self],
            ldv: i32,
            t: &[Self],
            ldt: i32,
            c: &mut [Self],
            ldc: i32,
            work: &mut [Self],
        ) -> i32;

        /// The workspace sizes the reference wrapper would have allocated for itself.
        fn geqrf_scratch(m: usize, n: usize) -> usize;
        fn orgqr_scratch(m: usize, n: usize, k: usize) -> usize;
        fn ormqr_scratch(m: usize, ncols: usize, k: usize) -> usize;
        /// `(work, rwork, iwork)` for `gesdd`.
        fn gesdd_scratch(m: usize, n: usize, thin: bool) -> (usize, usize, usize);
        /// `(work, rwork, iwork)` for `syevd`/`heevd`.
        fn evd_scratch(n: usize) -> (usize, usize, usize);
    }

    impl LapackBlas for f64 {
        fn trsm(layout: i32, n: usize, nrhs: usize, u: &[f64], ldu: i32, b: &mut [f64], ldb: i32) {
            // SAFETY: `u` holds the `n x n` triangle and `b` the `n x nrhs` right-hand side with
            // the leading dimensions passed; CBLAS reads only the upper triangle of `u`, writes
            // `b` in place and retains neither.
            unsafe {
                cblas_dtrsm(
                    layout,
                    CBLAS_LEFT,
                    CBLAS_UPPER,
                    CBLAS_NO_TRANS,
                    CBLAS_NON_UNIT,
                    n as i32,
                    nrhs as i32,
                    1.0,
                    u.as_ptr(),
                    ldu,
                    b.as_mut_ptr(),
                    ldb,
                );
            }
        }
        fn gemm(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &[f64],
            b: &[f64],
            c: &mut [f64],
            lda: i32,
            ldb: i32,
            ldc: i32,
        ) {
            // SAFETY: the buffers hold `m*k`, `k*n` and `m*n` elements with the
            // leading dimensions passed, and CBLAS does not retain them.
            unsafe {
                cblas_dgemm(
                    layout,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    lda,
                    b.as_ptr(),
                    ldb,
                    0.0,
                    c.as_mut_ptr(),
                    ldc,
                );
            }
        }
    }

    impl LapackScalar for f64 {
        fn getc2(
            layout: i32,
            n: usize,
            a: &mut [f64],
            lda: i32,
            ipiv: &mut [i32],
            jpiv: &mut [i32],
        ) -> i32 {
            // The Fortran routine is column-major by definition.
            assert_eq!(layout, COL_MAJOR, "?getc2 has no row-major entry point");
            let n = n as i32;
            let mut info = 0i32;
            // SAFETY: `a` is `n x n` at `lda`, `ipiv`/`jpiv` hold `n` entries each, and the
            // scalars are live locals for the duration of the call.
            unsafe {
                dgetc2_(
                    &n,
                    a.as_mut_ptr(),
                    &lda,
                    ipiv.as_mut_ptr(),
                    jpiv.as_mut_ptr(),
                    &mut info,
                );
            }
            info
        }
        fn larft(
            layout: i32,
            n: usize,
            k: usize,
            v: &[f64],
            ldv: i32,
            tau: &[f64],
            t: &mut [f64],
            ldt: i32,
        ) -> i32 {
            // SAFETY: `v` is `n x k` at `ldv`, `tau` holds `k` scalars, `t` is `k x k` at `ldt`.
            unsafe {
                LAPACKE_dlarft_work(
                    layout,
                    DIRECT_F,
                    STOREV_C,
                    n as i32,
                    k as i32,
                    v.as_ptr(),
                    ldv,
                    tau.as_ptr(),
                    t.as_mut_ptr(),
                    ldt,
                )
            }
        }
        fn larfb(
            layout: i32,
            m: usize,
            ncols: usize,
            k: usize,
            v: &[f64],
            ldv: i32,
            t: &[f64],
            ldt: i32,
            c: &mut [f64],
            ldc: i32,
            work: &mut [f64],
        ) -> i32 {
            debug_assert!(work.len() >= ncols.max(1) * k);
            // SAFETY: `v` is `m x k` at `ldv` (only its unit lower trapezoid is read), `t` is
            // `k x k` at `ldt`, `c` is `m x ncols` at `ldc`, and `work` is `ncols x k` at
            // `ldwork = ncols`, the documented minimum for the left side.
            unsafe {
                LAPACKE_dlarfb_work(
                    layout,
                    SIDE_L,
                    TRANS_N,
                    DIRECT_F,
                    STOREV_C,
                    m as i32,
                    ncols as i32,
                    k as i32,
                    v.as_ptr(),
                    ldv,
                    t.as_ptr(),
                    ldt,
                    c.as_mut_ptr(),
                    ldc,
                    work.as_mut_ptr(),
                    ncols.max(1) as i32,
                )
            }
        }
        fn as_len(self) -> usize {
            self as usize
        }

        fn geqrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [f64],
            lda: i32,
            tau: &mut [f64],
            work: &mut [f64],
        ) -> i32 {
            unsafe {
                LAPACKE_dgeqrf_work(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_mut_ptr(),
                    work.as_mut_ptr(),
                    work.len() as i32,
                )
            }
        }
        fn orgqr(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &mut [f64],
            lda: i32,
            tau: &[f64],
            work: &mut [f64],
        ) -> i32 {
            unsafe {
                LAPACKE_dorgqr_work(
                    layout,
                    m as i32,
                    n as i32,
                    k as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_ptr(),
                    work.as_mut_ptr(),
                    work.len() as i32,
                )
            }
        }
        fn ormqr(
            layout: i32,
            m: usize,
            ncols: usize,
            k: usize,
            a: &[f64],
            lda: i32,
            tau: &[f64],
            c: &mut [f64],
            ldc: i32,
            work: &mut [f64],
        ) -> i32 {
            unsafe {
                LAPACKE_dormqr_work(
                    layout,
                    SIDE_L,
                    TRANS_N,
                    m as i32,
                    ncols as i32,
                    k as i32,
                    a.as_ptr(),
                    lda,
                    tau.as_ptr(),
                    c.as_mut_ptr(),
                    ldc,
                    work.as_mut_ptr(),
                    work.len() as i32,
                )
            }
        }
        fn getrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [f64],
            lda: i32,
            ipiv: &mut [i32],
        ) -> i32 {
            unsafe {
                LAPACKE_dgetrf_work(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    ipiv.as_mut_ptr(),
                )
            }
        }
        fn potrf(layout: i32, n: usize, a: &mut [f64], lda: i32) -> i32 {
            unsafe { LAPACKE_dpotrf_work(layout, UPLO_L, n as i32, a.as_mut_ptr(), lda) }
        }
        fn gesv(
            layout: i32,
            n: usize,
            a: &mut [f64],
            lda: i32,
            ipiv: &mut [i32],
            b: &mut [f64],
            ldb: i32,
        ) -> i32 {
            unsafe {
                LAPACKE_dgesv_work(
                    layout,
                    n as i32,
                    1,
                    a.as_mut_ptr(),
                    lda,
                    ipiv.as_mut_ptr(),
                    b.as_mut_ptr(),
                    ldb,
                )
            }
        }
        #[allow(clippy::too_many_arguments)]
        fn gesdd(
            layout: i32,
            jobz: c_char,
            m: usize,
            n: usize,
            a: &mut [f64],
            lda: i32,
            s: &mut [f64],
            u: &mut [f64],
            ldu: i32,
            vt: &mut [f64],
            ldvt: i32,
            work: &mut [f64],
            _rwork: &mut [f64],
            iwork: &mut [i32],
        ) -> i32 {
            unsafe {
                LAPACKE_dgesdd_work(
                    layout,
                    jobz,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    s.as_mut_ptr(),
                    u.as_mut_ptr(),
                    ldu,
                    vt.as_mut_ptr(),
                    ldvt,
                    work.as_mut_ptr(),
                    work.len() as i32,
                    iwork.as_mut_ptr(),
                )
            }
        }
        fn evd(
            layout: i32,
            jobz: c_char,
            n: usize,
            a: &mut [f64],
            lda: i32,
            w: &mut [f64],
            work: &mut [f64],
            _rwork: &mut [f64],
            iwork: &mut [i32],
        ) -> i32 {
            unsafe {
                LAPACKE_dsyevd_work(
                    layout,
                    jobz,
                    UPLO_L,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    w.as_mut_ptr(),
                    work.as_mut_ptr(),
                    work.len() as i32,
                    iwork.as_mut_ptr(),
                    iwork.len() as i32,
                )
            }
        }

        fn geqrf_scratch(m: usize, n: usize) -> usize {
            let mut a = vec![0.0f64; m * n];
            let mut tau = vec![0.0f64; m.min(n)];
            let mut work = [0.0f64; 1];
            // SAFETY: `lwork = -1` is the documented size query; it writes the
            // optimum into `work[0]` and touches nothing else.
            let info = unsafe {
                LAPACKE_dgeqrf_work(
                    COL_MAJOR,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    leading(m, n, Order::F),
                    tau.as_mut_ptr(),
                    work.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "geqrf size query failed");
            work[0].as_len()
        }
        fn orgqr_scratch(m: usize, n: usize, k: usize) -> usize {
            let mut a = vec![0.0f64; m * n];
            let tau = vec![0.0f64; k];
            let mut work = [0.0f64; 1];
            let info = unsafe {
                LAPACKE_dorgqr_work(
                    COL_MAJOR,
                    m as i32,
                    n as i32,
                    k as i32,
                    a.as_mut_ptr(),
                    leading(m, n, Order::F),
                    tau.as_ptr(),
                    work.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "orgqr size query failed");
            work[0].as_len()
        }
        fn ormqr_scratch(m: usize, ncols: usize, k: usize) -> usize {
            let a = vec![0.0f64; m * k];
            let tau = vec![0.0f64; k];
            let mut c = vec![0.0f64; m * ncols];
            let mut work = [0.0f64; 1];
            let info = unsafe {
                LAPACKE_dormqr_work(
                    COL_MAJOR,
                    SIDE_L,
                    TRANS_N,
                    m as i32,
                    ncols as i32,
                    k as i32,
                    a.as_ptr(),
                    leading(m, k, Order::F),
                    tau.as_ptr(),
                    c.as_mut_ptr(),
                    leading(m, ncols, Order::F),
                    work.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "ormqr size query failed");
            work[0].as_len()
        }
        fn gesdd_scratch(m: usize, n: usize, thin: bool) -> (usize, usize, usize) {
            let r = m.min(n);
            let mut a = vec![0.0f64; m * n];
            let mut s = vec![0.0f64; r];
            let (mut u, mut vt) = if thin {
                (vec![0.0f64; m * r], vec![0.0f64; r * n])
            } else {
                (vec![0.0f64; 1], vec![0.0f64; 1])
            };
            // `gesdd` has no `liwork` parameter: the reference wrapper sizes `iwork`
            // by formula and has no `rwork` at all for the real dtype.
            let mut iwork = vec![0i32; 8 * r.max(1)];
            let mut work = [0.0f64; 1];
            let jobz = if thin { JOB_S } else { JOB_N };
            let info = unsafe {
                LAPACKE_dgesdd_work(
                    COL_MAJOR,
                    jobz,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    leading(m, n, Order::F),
                    s.as_mut_ptr(),
                    u.as_mut_ptr(),
                    ld_u(m, r, Order::F),
                    vt.as_mut_ptr(),
                    ld_vt(r, n, Order::F),
                    work.as_mut_ptr(),
                    -1,
                    iwork.as_mut_ptr(),
                )
            };
            debug_assert_eq!(info, 0, "gesdd size query failed");
            (work[0].as_len(), 0, iwork.len())
        }
        fn evd_scratch(n: usize) -> (usize, usize, usize) {
            let mut a = vec![0.0f64; n * n];
            let mut w = vec![0.0f64; n];
            let mut work = [0.0f64; 1];
            let mut iwork = [0i32; 1];
            let info = unsafe {
                LAPACKE_dsyevd_work(
                    COL_MAJOR,
                    JOB_V,
                    UPLO_L,
                    n as i32,
                    a.as_mut_ptr(),
                    leading(n, n, Order::F),
                    w.as_mut_ptr(),
                    work.as_mut_ptr(),
                    -1,
                    iwork.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "syevd size query failed");
            (work[0].as_len(), 0, iwork[0].max(1) as usize)
        }
    }

    impl LapackBlas for Complex64 {
        fn trsm(
            layout: i32,
            n: usize,
            nrhs: usize,
            u: &[Complex64],
            ldu: i32,
            b: &mut [Complex64],
            ldb: i32,
        ) {
            let alpha = Complex64::new(1.0, 0.0);
            // SAFETY: as for the real case; the complex ABI is the flat `{re, im}` struct.
            unsafe {
                cblas_ztrsm(
                    layout,
                    CBLAS_LEFT,
                    CBLAS_UPPER,
                    CBLAS_NO_TRANS,
                    CBLAS_NON_UNIT,
                    n as i32,
                    nrhs as i32,
                    (&alpha as *const Complex64).cast::<c_void>(),
                    u.as_ptr().cast::<c_void>(),
                    ldu,
                    b.as_mut_ptr().cast::<c_void>(),
                    ldb,
                );
            }
        }
        fn gemm(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &[Complex64],
            b: &[Complex64],
            c: &mut [Complex64],
            lda: i32,
            ldb: i32,
            ldc: i32,
        ) {
            let alpha = Complex64::new(1.0, 0.0);
            let beta = Complex64::new(0.0, 0.0);
            // SAFETY: as for the real case; the complex ABI is the flat struct MKL documents.
            unsafe {
                cblas_zgemm(
                    layout,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    (&alpha as *const Complex64).cast::<c_void>(),
                    a.as_ptr().cast::<c_void>(),
                    lda,
                    b.as_ptr().cast::<c_void>(),
                    ldb,
                    (&beta as *const Complex64).cast::<c_void>(),
                    c.as_mut_ptr().cast::<c_void>(),
                    ldc,
                );
            }
        }
    }

    impl LapackBlas for f32 {
        fn trsm(layout: i32, n: usize, nrhs: usize, u: &[f32], ldu: i32, b: &mut [f32], ldb: i32) {
            // SAFETY: `u` holds the `n x n` triangle and `b` the `n x nrhs` right-hand side with
            // the leading dimensions passed; CBLAS reads only the upper triangle of `u`, writes
            // `b` in place and retains neither.
            unsafe {
                cblas_strsm(
                    layout,
                    CBLAS_LEFT,
                    CBLAS_UPPER,
                    CBLAS_NO_TRANS,
                    CBLAS_NON_UNIT,
                    n as i32,
                    nrhs as i32,
                    1.0,
                    u.as_ptr(),
                    ldu,
                    b.as_mut_ptr(),
                    ldb,
                );
            }
        }
        fn gemm(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &[f32],
            b: &[f32],
            c: &mut [f32],
            lda: i32,
            ldb: i32,
            ldc: i32,
        ) {
            // SAFETY: the buffers hold `m*k`, `k*n` and `m*n` elements with the
            // leading dimensions passed, and CBLAS does not retain them.
            unsafe {
                cblas_sgemm(
                    layout,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    a.as_ptr(),
                    lda,
                    b.as_ptr(),
                    ldb,
                    0.0,
                    c.as_mut_ptr(),
                    ldc,
                );
            }
        }
    }

    impl LapackScalar for Complex64 {
        fn getc2(
            layout: i32,
            n: usize,
            a: &mut [Complex64],
            lda: i32,
            ipiv: &mut [i32],
            jpiv: &mut [i32],
        ) -> i32 {
            // The Fortran routine is column-major by definition.
            assert_eq!(layout, COL_MAJOR, "?getc2 has no row-major entry point");
            let n = n as i32;
            let mut info = 0i32;
            // SAFETY: `a` is `n x n` at `lda`, `ipiv`/`jpiv` hold `n` entries each, and the
            // scalars are live locals for the duration of the call.
            unsafe {
                zgetc2_(
                    &n,
                    a.as_mut_ptr(),
                    &lda,
                    ipiv.as_mut_ptr(),
                    jpiv.as_mut_ptr(),
                    &mut info,
                );
            }
            info
        }
        fn larft(
            layout: i32,
            n: usize,
            k: usize,
            v: &[Complex64],
            ldv: i32,
            tau: &[Complex64],
            t: &mut [Complex64],
            ldt: i32,
        ) -> i32 {
            // SAFETY: `v` is `n x k` at `ldv`, `tau` holds `k` scalars, `t` is `k x k` at `ldt`.
            unsafe {
                LAPACKE_zlarft_work(
                    layout,
                    DIRECT_F,
                    STOREV_C,
                    n as i32,
                    k as i32,
                    v.as_ptr(),
                    ldv,
                    tau.as_ptr(),
                    t.as_mut_ptr(),
                    ldt,
                )
            }
        }
        fn larfb(
            layout: i32,
            m: usize,
            ncols: usize,
            k: usize,
            v: &[Complex64],
            ldv: i32,
            t: &[Complex64],
            ldt: i32,
            c: &mut [Complex64],
            ldc: i32,
            work: &mut [Complex64],
        ) -> i32 {
            debug_assert!(work.len() >= ncols.max(1) * k);
            // SAFETY: `v` is `m x k` at `ldv` (only its unit lower trapezoid is read), `t` is
            // `k x k` at `ldt`, `c` is `m x ncols` at `ldc`, and `work` is `ncols x k` at
            // `ldwork = ncols`, the documented minimum for the left side.
            unsafe {
                LAPACKE_zlarfb_work(
                    layout,
                    SIDE_L,
                    TRANS_N,
                    DIRECT_F,
                    STOREV_C,
                    m as i32,
                    ncols as i32,
                    k as i32,
                    v.as_ptr(),
                    ldv,
                    t.as_ptr(),
                    ldt,
                    c.as_mut_ptr(),
                    ldc,
                    work.as_mut_ptr(),
                    ncols.max(1) as i32,
                )
            }
        }
        fn as_len(self) -> usize {
            self.re as usize
        }

        fn geqrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [Complex64],
            lda: i32,
            tau: &mut [Complex64],
            work: &mut [Complex64],
        ) -> i32 {
            unsafe {
                LAPACKE_zgeqrf_work(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_mut_ptr(),
                    work.as_mut_ptr(),
                    work.len() as i32,
                )
            }
        }
        fn orgqr(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &mut [Complex64],
            lda: i32,
            tau: &[Complex64],
            work: &mut [Complex64],
        ) -> i32 {
            unsafe {
                LAPACKE_zungqr_work(
                    layout,
                    m as i32,
                    n as i32,
                    k as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_ptr(),
                    work.as_mut_ptr(),
                    work.len() as i32,
                )
            }
        }
        fn ormqr(
            layout: i32,
            m: usize,
            ncols: usize,
            k: usize,
            a: &[Complex64],
            lda: i32,
            tau: &[Complex64],
            c: &mut [Complex64],
            ldc: i32,
            work: &mut [Complex64],
        ) -> i32 {
            unsafe {
                LAPACKE_zunmqr_work(
                    layout,
                    SIDE_L,
                    TRANS_N,
                    m as i32,
                    ncols as i32,
                    k as i32,
                    a.as_ptr(),
                    lda,
                    tau.as_ptr(),
                    c.as_mut_ptr(),
                    ldc,
                    work.as_mut_ptr(),
                    work.len() as i32,
                )
            }
        }
        fn getrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [Complex64],
            lda: i32,
            ipiv: &mut [i32],
        ) -> i32 {
            unsafe {
                LAPACKE_zgetrf_work(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    ipiv.as_mut_ptr(),
                )
            }
        }
        fn potrf(layout: i32, n: usize, a: &mut [Complex64], lda: i32) -> i32 {
            unsafe { LAPACKE_zpotrf_work(layout, UPLO_L, n as i32, a.as_mut_ptr(), lda) }
        }
        fn gesv(
            layout: i32,
            n: usize,
            a: &mut [Complex64],
            lda: i32,
            ipiv: &mut [i32],
            b: &mut [Complex64],
            ldb: i32,
        ) -> i32 {
            unsafe {
                LAPACKE_zgesv_work(
                    layout,
                    n as i32,
                    1,
                    a.as_mut_ptr(),
                    lda,
                    ipiv.as_mut_ptr(),
                    b.as_mut_ptr(),
                    ldb,
                )
            }
        }
        #[allow(clippy::too_many_arguments)]
        fn gesdd(
            layout: i32,
            jobz: c_char,
            m: usize,
            n: usize,
            a: &mut [Complex64],
            lda: i32,
            s: &mut [f64],
            u: &mut [Complex64],
            ldu: i32,
            vt: &mut [Complex64],
            ldvt: i32,
            work: &mut [Complex64],
            rwork: &mut [f64],
            iwork: &mut [i32],
        ) -> i32 {
            unsafe {
                LAPACKE_zgesdd_work(
                    layout,
                    jobz,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    s.as_mut_ptr(),
                    u.as_mut_ptr(),
                    ldu,
                    vt.as_mut_ptr(),
                    ldvt,
                    work.as_mut_ptr(),
                    work.len() as i32,
                    rwork.as_mut_ptr(),
                    iwork.as_mut_ptr(),
                )
            }
        }
        fn evd(
            layout: i32,
            jobz: c_char,
            n: usize,
            a: &mut [Complex64],
            lda: i32,
            w: &mut [f64],
            work: &mut [Complex64],
            rwork: &mut [f64],
            iwork: &mut [i32],
        ) -> i32 {
            unsafe {
                LAPACKE_zheevd_work(
                    layout,
                    jobz,
                    UPLO_L,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    w.as_mut_ptr(),
                    work.as_mut_ptr(),
                    work.len() as i32,
                    rwork.as_mut_ptr(),
                    rwork.len() as i32,
                    iwork.as_mut_ptr(),
                    iwork.len() as i32,
                )
            }
        }

        fn geqrf_scratch(m: usize, n: usize) -> usize {
            let mut a = vec![Complex64::zero(); m * n];
            let mut tau = vec![Complex64::zero(); m.min(n)];
            let mut work = [Complex64::zero(); 1];
            let info = unsafe {
                LAPACKE_zgeqrf_work(
                    COL_MAJOR,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    leading(m, n, Order::F),
                    tau.as_mut_ptr(),
                    work.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "geqrf size query failed");
            work[0].as_len()
        }
        fn orgqr_scratch(m: usize, n: usize, k: usize) -> usize {
            let mut a = vec![Complex64::zero(); m * n];
            let tau = vec![Complex64::zero(); k];
            let mut work = [Complex64::zero(); 1];
            let info = unsafe {
                LAPACKE_zungqr_work(
                    COL_MAJOR,
                    m as i32,
                    n as i32,
                    k as i32,
                    a.as_mut_ptr(),
                    leading(m, n, Order::F),
                    tau.as_ptr(),
                    work.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "ungqr size query failed");
            work[0].as_len()
        }
        fn ormqr_scratch(m: usize, ncols: usize, k: usize) -> usize {
            let a = vec![Complex64::zero(); m * k];
            let tau = vec![Complex64::zero(); k];
            let mut c = vec![Complex64::zero(); m * ncols];
            let mut work = [Complex64::zero(); 1];
            let info = unsafe {
                LAPACKE_zunmqr_work(
                    COL_MAJOR,
                    SIDE_L,
                    TRANS_N,
                    m as i32,
                    ncols as i32,
                    k as i32,
                    a.as_ptr(),
                    leading(m, k, Order::F),
                    tau.as_ptr(),
                    c.as_mut_ptr(),
                    leading(m, ncols, Order::F),
                    work.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "unmqr size query failed");
            work[0].as_len()
        }
        fn gesdd_scratch(m: usize, n: usize, thin: bool) -> (usize, usize, usize) {
            let r = m.min(n);
            let mut a = vec![Complex64::zero(); m * n];
            let mut s = vec![0.0f64; r];
            let (mut u, mut vt) = if thin {
                (
                    vec![Complex64::zero(); m * r],
                    vec![Complex64::zero(); r * n],
                )
            } else {
                (vec![Complex64::zero(); 1], vec![Complex64::zero(); 1])
            };
            // `gesdd` has no `liwork` parameter, and the complex driver needs an
            // `rwork` the `lwork = -1` query does not report: the reference wrapper
            // sizes both by formula (`lapacke_zgesdd.c:53-57, 66-72`), and so does this.
            let mut iwork = vec![0i32; 8 * r.max(1)];
            let lrwork = if thin {
                (r.max(1) * (5 * r + 7).max(2 * m.max(n) + 2 * r + 1)).max(1)
            } else {
                (7 * r).max(1)
            };
            let mut rwork = vec![0.0f64; lrwork];
            let mut work = [Complex64::zero(); 1];
            let jobz = if thin { JOB_S } else { JOB_N };
            let info = unsafe {
                LAPACKE_zgesdd_work(
                    COL_MAJOR,
                    jobz,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    leading(m, n, Order::F),
                    s.as_mut_ptr(),
                    u.as_mut_ptr(),
                    ld_u(m, r, Order::F),
                    vt.as_mut_ptr(),
                    ld_vt(r, n, Order::F),
                    work.as_mut_ptr(),
                    -1,
                    rwork.as_mut_ptr(),
                    iwork.as_mut_ptr(),
                )
            };
            debug_assert_eq!(info, 0, "gesdd size query failed");
            (work[0].as_len(), rwork.len(), iwork.len())
        }
        fn evd_scratch(n: usize) -> (usize, usize, usize) {
            let mut a = vec![Complex64::zero(); n * n];
            let mut w = vec![0.0f64; n];
            let mut work = [Complex64::zero(); 1];
            let mut rwork = [0.0f64; 1];
            let mut iwork = [0i32; 1];
            let info = unsafe {
                LAPACKE_zheevd_work(
                    COL_MAJOR,
                    JOB_V,
                    UPLO_L,
                    n as i32,
                    a.as_mut_ptr(),
                    leading(n, n, Order::F),
                    w.as_mut_ptr(),
                    work.as_mut_ptr(),
                    -1,
                    rwork.as_mut_ptr(),
                    -1,
                    iwork.as_mut_ptr(),
                    -1,
                )
            };
            debug_assert_eq!(info, 0, "heevd size query failed");
            (
                work[0].as_len(),
                rwork[0] as usize,
                iwork[0].max(1) as usize,
            )
        }
    }

    impl LapackBlas for num_complex::Complex32 {
        fn trsm(
            layout: i32,
            n: usize,
            nrhs: usize,
            u: &[num_complex::Complex32],
            ldu: i32,
            b: &mut [num_complex::Complex32],
            ldb: i32,
        ) {
            let alpha = num_complex::Complex32::new(1.0, 0.0);
            // SAFETY: as for the real case; the complex ABI is the flat `{re, im}` struct.
            unsafe {
                cblas_ctrsm(
                    layout,
                    CBLAS_LEFT,
                    CBLAS_UPPER,
                    CBLAS_NO_TRANS,
                    CBLAS_NON_UNIT,
                    n as i32,
                    nrhs as i32,
                    (&alpha as *const num_complex::Complex32).cast::<c_void>(),
                    u.as_ptr().cast::<c_void>(),
                    ldu,
                    b.as_mut_ptr().cast::<c_void>(),
                    ldb,
                );
            }
        }
        fn gemm(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &[num_complex::Complex32],
            b: &[num_complex::Complex32],
            c: &mut [num_complex::Complex32],
            lda: i32,
            ldb: i32,
            ldc: i32,
        ) {
            let alpha = num_complex::Complex32::new(1.0, 0.0);
            let beta = num_complex::Complex32::new(0.0, 0.0);
            // SAFETY: as above; `Complex32` is the flat `{re, im}` layout the C ABI takes.
            unsafe {
                cblas_cgemm(
                    layout,
                    CBLAS_NO_TRANS,
                    CBLAS_NO_TRANS,
                    m as i32,
                    n as i32,
                    k as i32,
                    (&alpha as *const num_complex::Complex32).cast::<c_void>(),
                    a.as_ptr().cast::<c_void>(),
                    lda,
                    b.as_ptr().cast::<c_void>(),
                    ldb,
                    (&beta as *const num_complex::Complex32).cast::<c_void>(),
                    c.as_mut_ptr().cast::<c_void>(),
                    ldc,
                );
            }
        }
    }

    /// The row name the CSV carries: derived from what was linked, not requested.
    pub fn row_name() -> &'static str {
        if cfg!(feature = "link-mkl") {
            "lapack-mkl"
        } else if cfg!(feature = "link-aocl") {
            "lapack-aocl"
        } else {
            "lapack-openblas"
        }
    }

    fn layout(order: Order) -> i32 {
        match order {
            Order::C => ROW_MAJOR,
            Order::F => COL_MAJOR,
        }
    }

    /// The layout a LAPACK driver is called with. Column-major only: LAPACKE's
    /// row-major path copies and transposes inside the call, so it is not a route
    /// this campaign measures (see the module comment, and `driver_cell`).
    fn driver_layout(order: Order) -> i32 {
        debug_assert_eq!(
            order,
            Order::F,
            "a vendor LAPACK driver is called column-major only"
        );
        COL_MAJOR
    }

    /// The leading dimension of an `m x n` buffer stored in `order`.
    fn leading(m: usize, n: usize, order: Order) -> i32 {
        match order {
            Order::C => n as i32,
            Order::F => m as i32,
        }
    }

    /// The families whose vendor route is a LAPACK driver, i.e. the ones that are
    /// column-major only. `gemm` and `triangular_solve` are not: their routes are CBLAS
    /// level-3 calls (`?gemm`, `?trsm`), whose row-major entry point moves no data and
    /// allocates nothing.
    const DRIVER_FAMILIES: &[&str] = &[
        "qr_factor",
        "qr",
        "lu",
        "full_piv_lu",
        "cholesky",
        "solve",
        "svd_values",
        "svd_thin",
        "eigh",
        "householder_apply",
    ];

    /// The families whose vendor call is a different *algorithm* from the faer rows' call,
    /// although it computes the same quantity under the same contract (in place, prepared
    /// buffers, no allocation inside the clock), so the rows are still put in one ratio.
    /// A route difference is a property to record, not to fix away
    /// (docs/reliable-measurement.md, "Route differences that remain"): `fbench info` prints
    /// each entry as `route_difference.<family>=<text>`, which the recorder keeps in the run's
    /// `binary-info.txt`, so a page can state it next to the ratio. The row's `note` stays the
    /// colmat row's own note: it is the join key (`KEY` in `scripts/gen_comparison.py`), and a
    /// different note would leave the cell without its vendor row instead of labelling it.
    pub const ROUTE_DIFFERENCES: &[(&str, &str)] = &[(
        "full_piv_lu",
        "vendor: unblocked ?getc2 (LAPACK ships no blocked complete-pivoting LU; square shapes \
         only, a non-square cell is skipped; c64 pivots on the modulus |z|); colmat/faer: \
         blocked full_piv_lu lu_in_place (c64 pivots on |re|+|im|)",
    )];

    fn within<T: Scalar>(residual: f64, scale: f64, size: usize, what: &str) -> Result<(), String> {
        let tol = tol_of::<T>(size) * scale.max(1.0);
        if residual.is_finite() && residual <= tol {
            Ok(())
        } else {
            Err(format!("{what} residual {residual:e} exceeds {tol:e}"))
        }
    }

    fn square(case: Case, family: &str) -> Result<(), String> {
        if case.m == case.n {
            Ok(())
        } else {
            Err(format!(
                "{family} needs a square matrix; {} is not",
                case.shape()
            ))
        }
    }

    /// The vendor rows for the two `f32` dtypes: the CBLAS routes (`?gemm`, `?trsm`) exist
    /// for both, the LAPACK drivers do not (in this harness), so every other family is a
    /// skip with the reason in it rather than a number or a dummy driver.
    fn ops_blas_only<T: LapackBlas>(
        dtype: Dtype,
        family: &str,
        case: Case,
        order: Order,
    ) -> Result<Vec<Op>, String> {
        match family {
            "gemm" => Ok(vec![gemm_op::<T>(
                case.m,
                case.k.unwrap_or(case.m),
                case.n,
                case.batch,
                order,
            )?]),
            "triangular_solve" => {
                square(case, family)?;
                Ok(vec![triangular_solve_op::<T>(case.m, case.batch, order)?])
            }
            other => Err(format!(
                "{other} has no {} LAPACK route in this harness yet; the vendor row is skipped",
                dtype.label()
            )),
        }
    }

    pub fn ops(dtype: Dtype, family: &str, case: Case, order: Order) -> Result<Vec<Op>, String> {
        match dtype {
            Dtype::F64 => ops_typed::<f64>(family, case, order),
            Dtype::C64 => ops_typed::<Complex64>(family, case, order),
            Dtype::F32 => ops_blas_only::<f32>(dtype, family, case, order),
            Dtype::C32 => ops_blas_only::<num_complex::Complex32>(dtype, family, case, order),
        }
    }

    /// A C-order driver cell has no route that leaves the clock free of allocation
    /// and layout conversion: LAPACKE's row-major path allocates a transposed copy
    /// and transposes in and out of it inside the call, in every `_work` variant.
    /// The faer rows factor a row-major input where it is, so a C-order vendor
    /// number would charge that conversion to one side only. It is recorded as a
    /// skip with this reason, never as a number.
    fn driver_cell(family: &str) -> String {
        if family == "full_piv_lu" {
            // `?getc2` has no LAPACKE wrapper at all: the Fortran routine is the only entry
            // point, and it is column-major by definition.
            return "full_piv_lu in c order has no vendor route: ?getc2 is a Fortran routine \
                    with no row-major entry point (LAPACKE does not wrap it), so a row-major \
                    input would need a transposed copy inside the vendor's clock. The vendor row \
                    is measured in f order, where both arms run column-major"
                .to_owned();
        }
        format!(
            "{family} in c order has no vendor route that leaves the clock free of allocation \
             and layout conversion: LAPACKE's row-major path allocates a transposed copy and \
             transposes in and out of it inside the call. The vendor row is measured in f \
             order, where both arms run column-major"
        )
    }

    fn ops_typed<T: LapackScalar + colmat::traits::ComplexField>(
        family: &str,
        case: Case,
        order: Order,
    ) -> Result<Vec<Op>, String> {
        if order == Order::C && DRIVER_FAMILIES.contains(&family) {
            return Err(driver_cell(family));
        }
        let m = case.m;
        let n = case.n;
        let batch = case.batch;
        match family {
            "gemm" => Ok(vec![gemm_op::<T>(m, case.k.unwrap_or(m), n, batch, order)?]),
            "qr_factor" => Ok(vec![qr_factor_op::<T>(m, n, batch, order)?]),
            // The composite `qr` (explicit Q) is "factor in place, form Q into a
            // prepared buffer" on both arms: `geqrf` + `ormqr` here,
            // `qr_in_place` + the block-Householder apply on the faer rows.
            "qr" => Ok(vec![qr_op::<T>(m, n, batch, order)?]),
            "lu" => Ok(vec![lu_op::<T>(m, n, batch, order)?]),
            "cholesky" => {
                square(case, family)?;
                Ok(vec![cholesky_op::<T>(m, batch, order)?])
            }
            "solve" => {
                square(case, family)?;
                Ok(vec![solve_op::<T>(m, batch, order)?])
            }
            "svd_values" => Ok(vec![svd_op::<T>(m, n, batch, order, false)?]),
            "svd_thin" => Ok(vec![svd_op::<T>(m, n, batch, order, true)?]),
            "eigh" => {
                square(case, family)?;
                Ok(vec![eigh_op::<T>(m, batch, order)?])
            }
            "triangular_solve" => {
                square(case, family)?;
                Ok(vec![triangular_solve_op::<T>(m, batch, order)?])
            }
            // `?getc2` is the only complete-pivoting LU LAPACK ships, and it is square only;
            // the faer rows take any shape, so a non-square cell has no vendor counterpart and is
            // skipped with that reason, never measured through some other call.
            "full_piv_lu" => {
                if m != n {
                    return Err(format!(
                        "full_piv_lu has no vendor route for the non-square shape {}: ?getc2, the \
                         only complete-pivoting LU LAPACK ships, is square only; the vendor row \
                         is skipped",
                        case.shape()
                    ));
                }
                Ok(vec![full_piv_lu_op::<T>(m, batch, order)?])
            }
            // The two public forms the faer rows time, with the same notes so the join pairs
            // block with block and sequence with sequence: `?larfb` applies the block reflector
            // `I - V T V^H`, `?ormqr`/`?unmqr` the reflector sequence.
            "householder_apply" => Ok(vec![
                householder_op::<T>(m, n, batch, order, false)?,
                householder_op::<T>(m, n, batch, order, true)?,
            ]),
            other => Err(format!(
                "{other} has no LAPACK route in this harness; the vendor row is skipped"
            )),
        }
    }

    fn gemm_op<T: LapackBlas>(
        m: usize,
        k: usize,
        n: usize,
        batch: usize,
        order: Order,
    ) -> Result<Op, String> {
        let a = fill_seeded::<T>(m * k * batch, seed(m, k, n));
        let b = fill_seeded::<T>(k * n * batch, seed(k, n, m));
        let mut c_ref = vec![T::zero(); m * n * batch];
        for bi in 0..batch {
            mul(
                m,
                k,
                n,
                order,
                &a[bi * m * k..(bi + 1) * m * k],
                &b[bi * k * n..(bi + 1) * k * n],
                &mut c_ref[bi * m * n..(bi + 1) * m * n],
            );
        }
        struct S<T> {
            a: Vec<T>,
            b: Vec<T>,
            c: Vec<T>,
            c_ref: Vec<T>,
            m: usize,
            k: usize,
            n: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            a,
            b,
            c: vec![T::zero(); m * n * batch],
            c_ref,
            m,
            k,
            n,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            |s| {
                s.c.fill(T::zero());
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    T::gemm(
                        layout(s.order),
                        s.m,
                        s.n,
                        s.k,
                        &s.a[bi * s.m * s.k..],
                        &s.b[bi * s.k * s.n..],
                        &mut s.c[bi * s.m * s.n..],
                        leading(s.m, s.k, s.order),
                        leading(s.k, s.n, s.order),
                        leading(s.m, s.n, s.order),
                    );
                }
                Ok(())
            },
            |s| {
                within::<T>(
                    max_diff(&s.c, &s.c_ref),
                    frob(&s.c_ref),
                    s.n.max(s.m),
                    "gemm",
                )
            },
        ))
    }

    /// The `m x r` `Q` of a compact `geqrf` factor, built outside the clock.
    fn thin_q<T: LapackScalar>(
        factor: &[T],
        tau: &[T],
        m: usize,
        n: usize,
        order: Order,
        work: &mut [T],
    ) -> Result<Vec<T>, String> {
        let r = m.min(n);
        let mut a = factor.to_vec();
        // Q is m x r with r = min(m, n) columns; asking orgqr for more columns than Q
        // has (n > m on a wide shape) is invalid and MKL rejects it.
        let info = T::orgqr(
            driver_layout(order),
            m,
            r,
            r,
            &mut a,
            leading(m, n, order),
            tau,
            work,
        );
        if info != 0 {
            return Err(format!("orgqr returned {info}"));
        }
        let mut qmat = vec![T::zero(); m * r];
        for i in 0..m {
            for j in 0..r {
                qmat[idx(i, j, m, r, order)] = a[idx(i, j, m, n, order)];
            }
        }
        Ok(qmat)
    }

    /// The `r x n` `R` of a compact factor: its upper trapezoid.
    fn upper_r<T: Scalar>(factor: &[T], m: usize, n: usize, order: Order) -> Vec<T> {
        let r = m.min(n);
        let mut rmat = vec![T::zero(); r * n];
        for i in 0..r {
            for j in i..n {
                rmat[idx(i, j, r, n, order)] = factor[idx(i, j, m, n, order)];
            }
        }
        rmat
    }

    /// `qr_factor` times `geqrf_work` on the prepared buffer. The check derives Q with
    /// `orgqr` from the ACTUAL compact factors (outside the clock) and reconstructs
    /// `A = Q R`.
    fn qr_factor_op<T: LapackScalar>(
        m: usize,
        n: usize,
        batch: usize,
        order: Order,
    ) -> Result<Op, String> {
        let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, m.min(n)));
        // Q is m x r with r = min(m, n) columns, and `orgqr` requires `n <= m`: the size query has
        // to ask for Q's own shape, the same rule `thin_q` follows for the call itself. Asking it
        // for the factor's full column count is invalid on a wide shape, and the vendor answers a
        // rejected query by printing a diagnostic to stdout (MKL: `Parameter 2 was incorrect on
        // entry to DORGQR`; OpenBLAS: its xerbla line), which the cell child's protocol rejects as
        // a line it does not define -- the row is then recorded `failed` on a correct measurement.
        let r = m.min(n);
        let lwork = T::geqrf_scratch(m, n).max(T::orgqr_scratch(m, r, r));
        struct S<T> {
            a0: Vec<T>,
            a: Vec<Vec<T>>,
            tau: Vec<Vec<T>>,
            work: Vec<T>,
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
        }
        // Every buffer the vendor call writes is the harness's, so it is allocated here, once,
        // and reused by every call (docs/reliable-measurement.md, "Allocation and the clock").
        let state = S {
            a: a0.chunks_exact(m * n).map(|a| a.to_vec()).collect(),
            tau: vec![vec![T::zero(); m.min(n)]; batch],
            work: vec![T::zero(); lwork.max(1)],
            a0,
            m,
            n,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            // The buffers are reused, so prepare restores the input and spoils the retained tau
            // (outside the clock): a check can only pass on what the next timed call writes.
            |s| {
                for bi in 0..s.batch {
                    s.a[bi].copy_from_slice(&s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n]);
                    s.tau[bi].fill(T::zero());
                }
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    let info = T::geqrf(
                        driver_layout(s.order),
                        s.m,
                        s.n,
                        &mut s.a[bi],
                        leading(s.m, s.n, s.order),
                        &mut s.tau[bi],
                        &mut s.work,
                    );
                    if info != 0 {
                        return Err(format!("geqrf returned {info}"));
                    }
                }
                Ok(())
            },
            |s| {
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                    let rmat = upper_r(&s.a[bi], s.m, s.n, s.order);
                    let qmat = thin_q(&s.a[bi], &s.tau[bi], s.m, s.n, s.order, &mut s.work)?;
                    let mut recon = vec![T::zero(); s.m * s.n];
                    mul(s.m, s.m.min(s.n), s.n, s.order, &qmat, &rmat, &mut recon);
                    within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "qr_factor")?;
                    within::<T>(
                        gram_residual(&qmat, s.m, s.m.min(s.n), s.order),
                        1.0,
                        s.m,
                        "qr_factor Q",
                    )?;
                }
                Ok(())
            },
        ))
    }

    /// `qr` times `geqrf_work` followed by `ormqr_work`: the factorisation in place,
    /// then `Q` formed into a prepared `m x min(m, n)` buffer. `R` is the factor
    /// buffer's upper trapezoid, which `ormqr` only reads, so both are the direct
    /// outputs of the two calls — the same contract the faer rows time
    /// (`qr_in_place` plus the block-Householder apply into a prepared `Q`).
    fn qr_op<T: LapackScalar>(
        m: usize,
        n: usize,
        batch: usize,
        order: Order,
    ) -> Result<Op, String> {
        let r = m.min(n);
        let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 3));
        let lwork = T::geqrf_scratch(m, n).max(T::ormqr_scratch(m, r, r));
        struct S<T> {
            a0: Vec<T>,
            a: Vec<Vec<T>>,
            tau: Vec<Vec<T>>,
            q: Vec<Vec<T>>,
            work: Vec<T>,
            m: usize,
            n: usize,
            size: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            a: a0.chunks_exact(m * n).map(|a| a.to_vec()).collect(),
            tau: vec![vec![T::zero(); r]; batch],
            q: vec![vec![T::zero(); m * r]; batch],
            work: vec![T::zero(); lwork.max(1)],
            a0,
            m,
            n,
            size: r,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            |s| {
                for bi in 0..s.batch {
                    s.a[bi].copy_from_slice(&s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n]);
                    s.tau[bi].fill(T::zero());
                    s.q[bi].fill(T::zero());
                    for i in 0..s.size {
                        s.q[bi][idx(i, i, s.m, s.size, s.order)] = T::one();
                    }
                }
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    let info = T::geqrf(
                        driver_layout(s.order),
                        s.m,
                        s.n,
                        &mut s.a[bi],
                        leading(s.m, s.n, s.order),
                        &mut s.tau[bi],
                        &mut s.work,
                    );
                    if info != 0 {
                        return Err(format!("geqrf returned {info}"));
                    }
                    // Q = (the k reflectors) applied to the prepared identity block.
                    let info = T::ormqr(
                        driver_layout(s.order),
                        s.m,
                        s.size,
                        s.size,
                        &s.a[bi],
                        leading(s.m, s.n, s.order),
                        &s.tau[bi],
                        &mut s.q[bi],
                        leading(s.m, s.size, s.order),
                        &mut s.work,
                    );
                    if info != 0 {
                        return Err(format!("ormqr returned {info}"));
                    }
                }
                Ok(())
            },
            |s| {
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                    let rmat = upper_r(&s.a[bi], s.m, s.n, s.order);
                    let qmat = &s.q[bi];
                    let mut recon = vec![T::zero(); s.m * s.n];
                    mul(s.m, s.size, s.n, s.order, qmat, &rmat, &mut recon);
                    within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "qr")?;
                    within::<T>(gram_residual(qmat, s.m, s.size, s.order), 1.0, s.m, "qr Q")?;
                }
                Ok(())
            },
        ))
    }

    /// `lu` times `getrf_work`; the check rebuilds `L` and `U` from the factors and
    /// compares `P A` against `L U`, which is the factorization's own contract
    /// for any shape (`L` is `m x min(m, n)` unit-lower trapezoidal, `U` is
    /// `min(m, n) x n` upper trapezoidal).
    fn lu_op<T: LapackScalar>(
        m: usize,
        n: usize,
        batch: usize,
        order: Order,
    ) -> Result<Op, String> {
        let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 5));
        let size = m.min(n);
        struct S<T> {
            a0: Vec<T>,
            outputs: Vec<(Vec<T>, Vec<i32>)>,
            m: usize,
            n: usize,
            size: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            outputs: (0..batch)
                .map(|_| (vec![T::zero(); m * n], vec![0i32; size]))
                .collect(),
            a0,
            m,
            n,
            size,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            // prepare restores the input into the reused buffer (outside the clock), so the timed
            // call is the in-place factorization alone, as the faer rows' `in_place` op.
            |s| {
                for (bi, (a, ipiv)) in s.outputs.iter_mut().enumerate() {
                    a.copy_from_slice(&s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n]);
                    ipiv.fill(0);
                }
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    let (a, ipiv) = &mut s.outputs[bi];
                    let info = T::getrf(
                        driver_layout(s.order),
                        s.m,
                        s.n,
                        a,
                        leading(s.m, s.n, s.order),
                        ipiv,
                    );
                    if info != 0 {
                        return Err(format!("getrf returned {info}"));
                    }
                }
                Ok(())
            },
            |s| {
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                    let (a, ipiv) = &s.outputs[bi];
                    // L is the lower trapezoid of `a` with a unit diagonal, U its
                    // upper trapezoid.
                    let mut l = vec![T::zero(); s.m * s.size];
                    let mut u = vec![T::zero(); s.size * s.n];
                    for i in 0..s.m {
                        for j in 0..s.size {
                            if i > j {
                                l[idx(i, j, s.m, s.size, s.order)] =
                                    a[idx(i, j, s.m, s.n, s.order)];
                            } else if i == j {
                                l[idx(i, j, s.m, s.size, s.order)] = T::one();
                            }
                        }
                    }
                    for i in 0..s.size {
                        for j in i..s.n {
                            u[idx(i, j, s.size, s.n, s.order)] = a[idx(i, j, s.m, s.n, s.order)];
                        }
                    }
                    let mut lu = vec![T::zero(); s.m * s.n];
                    mul(s.m, s.size, s.n, s.order, &l, &u, &mut lu);
                    // And `P A`: the same row swaps `getrf` chose, applied to the
                    // input. `ipiv` is one-based.
                    let mut pa = a0.to_vec();
                    for k in 0..s.size {
                        if ipiv[k] < 1 || ipiv[k] as usize > s.m {
                            return Err("getrf returned an invalid pivot".to_owned());
                        }
                        let t = (ipiv[k] - 1) as usize;
                        if t != k {
                            for j in 0..s.n {
                                pa.swap(idx(k, j, s.m, s.n, s.order), idx(t, j, s.m, s.n, s.order));
                            }
                        }
                    }
                    within::<T>(max_diff(&pa, &lu), frob(&pa), s.n.max(s.m), "lu")?;
                }
                Ok(())
            },
        ))
    }

    fn cholesky_op<T: LapackScalar>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
        let a0 = spd::<T>(n, batch, order, seed(n, n, 11));
        struct S<T> {
            a0: Vec<T>,
            outputs: Vec<Vec<T>>,
            n: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            outputs: vec![vec![T::zero(); n * n]; batch],
            a0,
            n,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            // prepare restores the input into the reused buffer (outside the clock), so the timed
            // call is the in-place factorization alone, as the faer rows' `in_place` op.
            |s| {
                for (bi, a) in s.outputs.iter_mut().enumerate() {
                    a.copy_from_slice(&s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n]);
                }
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    let a = &mut s.outputs[bi];
                    let info = T::potrf(driver_layout(s.order), s.n, a, leading(s.n, s.n, s.order));
                    if info != 0 {
                        return Err(format!("potrf returned {info}"));
                    }
                }
                Ok(())
            },
            |s| {
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                    let a = &s.outputs[bi];
                    // L is the lower triangle potrf left behind.
                    let mut l = vec![T::zero(); s.n * s.n];
                    for j in 0..s.n {
                        for i in j..s.n {
                            l[idx(i, j, s.n, s.n, s.order)] = a[idx(i, j, s.n, s.n, s.order)];
                        }
                    }
                    let mut recon = vec![T::zero(); s.n * s.n];
                    mul_adj_b(s.n, s.n, s.n, s.order, &l, &l, &mut recon);
                    within::<T>(max_diff(a0, &recon), frob(a0), s.n, "cholesky")?;
                }
                Ok(())
            },
        ))
    }

    fn solve_op<T: LapackScalar>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
        let a0 = diag_dominant::<T>(n, batch, order, seed(n, n, 13));
        let b0 = fill_seeded::<T>(n * batch, seed(n, 1, 17));
        struct S<T> {
            a0: Vec<T>,
            work: Vec<(Vec<T>, Vec<i32>)>,
            b0: Vec<T>,
            b: Vec<T>,
            sink: T,
            n: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            work: (0..batch)
                .map(|_| (vec![T::zero(); n * n], vec![0i32; n]))
                .collect(),
            a0,
            b0: b0.clone(),
            b: b0.clone(),
            sink: T::zero(),
            n,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            |s| {
                s.b.copy_from_slice(&s.b0);
                for (bi, (a, ipiv)) in s.work.iter_mut().enumerate() {
                    a.copy_from_slice(&s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n]);
                    ipiv.fill(0);
                }
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    let (a, ipiv) = &mut s.work[bi];
                    let info = T::gesv(
                        driver_layout(s.order),
                        s.n,
                        a,
                        leading(s.n, s.n, s.order),
                        ipiv,
                        &mut s.b[bi * s.n..],
                        leading(s.n, 1, s.order),
                    );
                    if info != 0 {
                        return Err(format!("gesv returned {info}"));
                    }
                    s.sink.add_assign(s.b[bi * s.n]);
                }
                Ok(())
            },
            |s| {
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                    let b0 = &s.b0[bi * s.n..(bi + 1) * s.n];
                    let x = &s.b[bi * s.n..(bi + 1) * s.n];
                    let mut ax = vec![T::zero(); s.n];
                    mul(s.n, s.n, 1, s.order, a0, x, &mut ax);
                    within::<T>(max_diff(&ax, b0), frob(b0), s.n, "solve")?;
                }
                Ok(())
            },
        ))
    }

    /// `svd_values` times `gesdd_work` with `jobz='N'`; `svd_thin` with `jobz='S'`.
    fn svd_op<T: LapackScalar>(
        m: usize,
        n: usize,
        batch: usize,
        order: Order,
        thin: bool,
    ) -> Result<Op, String> {
        let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, if thin { 31 } else { 29 }));
        let (lwork, lrwork, liwork) = T::gesdd_scratch(m, n, thin);
        struct S<T> {
            a0: Vec<T>,
            work: Vec<Vec<T>>,
            outputs: Vec<(Vec<f64>, Vec<T>, Vec<T>)>,
            ws: Vec<T>,
            rwork: Vec<f64>,
            iwork: Vec<i32>,
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
            thin: bool,
        }
        let r = m.min(n);
        let state = S {
            work: vec![vec![T::zero(); m * n]; batch],
            outputs: (0..batch)
                .map(|_| {
                    let (u, vt) = if thin { (m * r, r * n) } else { (0, 0) };
                    (vec![0.0f64; r], vec![T::zero(); u], vec![T::zero(); vt])
                })
                .collect(),
            ws: vec![T::zero(); lwork.max(1)],
            rwork: vec![0.0f64; lrwork.max(1)],
            iwork: vec![0i32; liwork.max(1)],
            a0,
            m,
            n,
            batch,
            order,
            thin,
        };
        Ok(op_from(
            state,
            String::new(),
            // `gesdd` overwrites its input, so prepare restores it into the prepared work buffer
            // (outside the clock, as for `getrf`): the timed call is `gesdd` alone, the call the faer
            // `in_place` rows make on a read-only input. The buffers are reused, so prepare also spoils
            // the retained result, and a check can only pass on what the next timed call writes.
            |s| {
                for bi in 0..s.batch {
                    s.work[bi].copy_from_slice(&s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n]);
                }
                for (sigma, u, _) in &mut s.outputs {
                    sigma[0] = -1.0;
                    if let Some(x) = u.first_mut() {
                        *x = T::from_f64(1e30);
                    }
                }
                for v in &mut s.iwork {
                    *v = 0;
                }
                Ok(())
            },
            |s, _tk| {
                let r = s.m.min(s.n);
                for bi in 0..s.batch {
                    let a = &mut s.work[bi];
                    let (sigma, u, vt) = &mut s.outputs[bi];
                    let mut dummy_u = [T::zero()];
                    let mut dummy_vt = [T::zero()];
                    let jobz = if s.thin { JOB_S } else { JOB_N };
                    let info = T::gesdd(
                        driver_layout(s.order),
                        jobz,
                        s.m,
                        s.n,
                        a,
                        leading(s.m, s.n, s.order),
                        sigma,
                        if s.thin {
                            u.as_mut_slice()
                        } else {
                            &mut dummy_u
                        },
                        // LAPACKE validates leading dimensions even for JOB_N,
                        // although U and VT are not referenced in that mode.
                        ld_u(s.m, r, s.order),
                        if s.thin {
                            vt.as_mut_slice()
                        } else {
                            &mut dummy_vt
                        },
                        ld_vt(r, s.n, s.order),
                        &mut s.ws,
                        &mut s.rwork,
                        &mut s.iwork,
                    );
                    if info != 0 {
                        return Err(format!("gesdd returned {info}"));
                    }
                }
                Ok(())
            },
            |s| {
                let r = s.m.min(s.n);
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                    let (sigma, u, vt) = &s.outputs[bi];
                    let eps = tol_of::<T>(s.n.max(s.m));
                    for i in 0..r {
                        if !sigma[i].is_finite() || sigma[i] < -eps {
                            return Err(format!("singular value {i} is negative: {}", sigma[i]));
                        }
                        if i + 1 < r && sigma[i] < sigma[i + 1] - eps {
                            return Err("singular values are not sorted".to_owned());
                        }
                    }
                    let sum: f64 = sigma.iter().map(|x| x * x).sum();
                    let scale = frob(a0);
                    within::<T>(
                        (sum.sqrt() - scale).abs(),
                        scale,
                        s.n.max(s.m),
                        "svd_values norm identity",
                    )?;
                    if s.thin {
                        let uu = &u[..s.m * r];
                        let vv = &vt[..r * s.n];
                        within::<T>(gram_residual(uu, s.m, r, s.order), 1.0, s.m, "svd_thin U")?;
                        within::<T>(gram_adj_residual(vv, r, s.n, s.order), 1.0, r, "svd_thin V")?;
                        let s_scale = scale_cols(
                            uu,
                            s.m,
                            r,
                            &sigma.iter().map(|x| T::from_f64(*x)).collect::<Vec<_>>(),
                            s.order,
                        );
                        let mut recon = vec![T::zero(); s.m * s.n];
                        mul(s.m, r, s.n, s.order, &s_scale, vv, &mut recon);
                        within::<T>(max_diff(a0, &recon), scale, s.n.max(s.m), "svd_thin")?;
                    }
                }
                Ok(())
            },
        ))
    }

    fn ld_u(m: usize, r: usize, order: Order) -> i32 {
        match order {
            Order::C => r as i32,
            Order::F => m as i32,
        }
    }

    fn ld_vt(r: usize, n: usize, order: Order) -> i32 {
        match order {
            Order::C => n as i32,
            Order::F => r as i32,
        }
    }

    fn eigh_op<T: LapackScalar>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
        let a0 = hermitian::<T>(n, order, seed(n, n, 37));
        let a0 = replicate(&a0, batch);
        let (lwork, lrwork, liwork) = T::evd_scratch(n);
        struct S<T> {
            a0: Vec<T>,
            outputs: Vec<(Vec<T>, Vec<f64>)>,
            ws: Vec<T>,
            rwork: Vec<f64>,
            iwork: Vec<i32>,
            n: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            outputs: (0..batch)
                .map(|_| (vec![T::zero(); n * n], vec![0.0f64; n]))
                .collect(),
            ws: vec![T::zero(); lwork.max(1)],
            rwork: vec![0.0f64; lrwork.max(1)],
            iwork: vec![0i32; liwork.max(1)],
            a0,
            n,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            // `syevd`/`heevd` overwrite their input with the eigenvectors, so prepare restores it
            // (outside the clock): the timed call is the driver alone, as the faer `in_place` row's
            // is `self_adjoint_evd` on a prepared `U`. Prepare also spoils the eigenvalues, so a check
            // can only pass on what the next timed call writes.
            |s| {
                for bi in 0..s.batch {
                    let (a, w) = &mut s.outputs[bi];
                    a.copy_from_slice(&s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n]);
                    w[0] = -1e30;
                }
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    let (a, w) = &mut s.outputs[bi];
                    let info = T::evd(
                        driver_layout(s.order),
                        JOB_V,
                        s.n,
                        a,
                        leading(s.n, s.n, s.order),
                        w,
                        &mut s.ws,
                        &mut s.rwork,
                        &mut s.iwork,
                    );
                    if info != 0 {
                        return Err(format!("syevd/heevd returned {info}"));
                    }
                }
                Ok(())
            },
            |s| {
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                    let (u, w) = &s.outputs[bi];
                    let scale = frob(a0);
                    within::<T>(gram_residual(u, s.n, s.n, s.order), 1.0, s.n, "eigh U")?;
                    let ws = w.iter().map(|x| T::from_f64(*x)).collect::<Vec<_>>();
                    let mut au = vec![T::zero(); s.n * s.n];
                    mul(s.n, s.n, s.n, s.order, a0, u, &mut au);
                    let us = scale_cols(u, s.n, s.n, &ws, s.order);
                    within::<T>(max_diff(&au, &us), scale, s.n, "eigh")?;
                }
                Ok(())
            },
        ))
    }

    /// `triangular_solve` times `cblas_?trsm` on the prepared right-hand side: the same input
    /// (`upper_triangular`, the same seeds) and the same contract as the faer rows'
    /// `solve_upper_triangular_in_place` -- one `n x 1` column, solved in place, no workspace.
    /// CBLAS's row-major entry point moves no data, so both orders are measured.
    fn triangular_solve_op<T: LapackBlas>(
        n: usize,
        batch: usize,
        order: Order,
    ) -> Result<Op, String> {
        let u0 = upper_triangular::<T>(n, batch, order, seed(n, n, 19));
        let b0 = fill_seeded::<T>(n * batch, seed(n, 1, 23));
        struct S<T> {
            u0: Vec<T>,
            b0: Vec<T>,
            b: Vec<T>,
            n: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            u0,
            b: b0.clone(),
            b0,
            n,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            // `?trsm` overwrites the right-hand side with the solution, so prepare restores it
            // (outside the clock): a check can only pass on what the next timed call wrote.
            |s| {
                s.b.copy_from_slice(&s.b0);
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    T::trsm(
                        layout(s.order),
                        s.n,
                        1,
                        &s.u0[bi * s.n * s.n..(bi + 1) * s.n * s.n],
                        leading(s.n, s.n, s.order),
                        &mut s.b[bi * s.n..(bi + 1) * s.n],
                        leading(s.n, 1, s.order),
                    );
                }
                Ok(())
            },
            |s| {
                for bi in 0..s.batch {
                    let u0 = &s.u0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                    let b0 = &s.b0[bi * s.n..(bi + 1) * s.n];
                    let x = &s.b[bi * s.n..(bi + 1) * s.n];
                    let mut ux = vec![T::zero(); s.n];
                    mul(s.n, s.n, 1, s.order, u0, x, &mut ux);
                    within::<T>(max_diff(&ux, b0), frob(b0), s.n, "triangular_solve")?;
                }
                Ok(())
            },
        ))
    }

    /// The forward permutation a sequence of LAPACK interchanges describes: entry `i` of the
    /// result is the original index that ended up at position `i` after swapping `i` with
    /// `piv[i] - 1` for `i = 0, 1, ...` in order (`?getc2`'s `ipiv`/`jpiv`, 1-based).
    fn interchanges_to_perm(piv: &[i32]) -> Result<Vec<usize>, String> {
        let n = piv.len();
        let mut perm: Vec<usize> = (0..n).collect();
        for (i, &p) in piv.iter().enumerate() {
            if p < 1 || p as usize > n {
                return Err(format!("interchange {i} names {p}, outside 1..={n}"));
            }
            perm.swap(i, p as usize - 1);
        }
        Ok(perm)
    }

    /// `full_piv_lu` times `?getc2` on the prepared buffer: LU with complete pivoting, in
    /// place, with the faer rows' input (the same seed). It is the **unblocked** LAPACK routine
    /// -- no vendor ships a blocked complete-pivoting LU -- against the faer rows' blocked
    /// `lu_in_place`, a route difference recorded in [`ROUTE_DIFFERENCES`]; square shapes only.
    /// The check rebuilds `P A Q` from the two interchange sequences and compares it with `L U`,
    /// the factorization's own contract and the same residual the faer rows check.
    fn full_piv_lu_op<T: LapackScalar>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
        let a0 = fill_seeded::<T>(n * n * batch, seed(n, n, 7));
        struct S<T> {
            a0: Vec<T>,
            outputs: Vec<(Vec<T>, Vec<i32>, Vec<i32>)>,
            n: usize,
            order: Order,
        }
        let state = S {
            outputs: (0..batch)
                .map(|_| (vec![T::zero(); n * n], vec![0i32; n], vec![0i32; n]))
                .collect(),
            a0,
            n,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            // prepare restores the input and spoils both interchange vectors (outside the clock),
            // so a check can only pass on what the next timed call writes.
            |s| {
                for (bi, (a, ipiv, jpiv)) in s.outputs.iter_mut().enumerate() {
                    a.copy_from_slice(&s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n]);
                    ipiv.fill(0);
                    jpiv.fill(0);
                }
                Ok(())
            },
            |s, _tk| {
                for (a, ipiv, jpiv) in s.outputs.iter_mut() {
                    let info = T::getc2(
                        driver_layout(s.order),
                        s.n,
                        a,
                        leading(s.n, s.n, s.order),
                        ipiv,
                        jpiv,
                    );
                    // `info > 0` means a pivot fell below `SMIN` and was perturbed: the factors
                    // then belong to a perturbed matrix, which is not the measured contract.
                    if info != 0 {
                        return Err(format!("getc2 returned {info}"));
                    }
                }
                Ok(())
            },
            |s| {
                let n = s.n;
                for (bi, (lu, ipiv, jpiv)) in s.outputs.iter().enumerate() {
                    let a0 = &s.a0[bi * n * n..(bi + 1) * n * n];
                    let rows = interchanges_to_perm(ipiv)?;
                    let cols = interchanges_to_perm(jpiv)?;
                    let mut l = vec![T::zero(); n * n];
                    let mut u = vec![T::zero(); n * n];
                    let mut paq = vec![T::zero(); n * n];
                    for i in 0..n {
                        for j in 0..n {
                            let v = lu[idx(i, j, n, n, s.order)];
                            if i > j {
                                l[idx(i, j, n, n, s.order)] = v;
                            } else {
                                u[idx(i, j, n, n, s.order)] = v;
                            }
                            paq[idx(i, j, n, n, s.order)] =
                                a0[idx(rows[i], cols[j], n, n, s.order)];
                        }
                        l[idx(i, i, n, n, s.order)] = T::one();
                    }
                    let mut prod = vec![T::zero(); n * n];
                    mul(n, n, n, s.order, &l, &u, &mut prod);
                    within::<T>(max_diff(&paq, &prod), frob(a0), n, "full_piv_lu")?;
                }
                Ok(())
            },
        ))
    }

    /// The `householder_apply` fixture of one batch entry: the `?geqrf`-layout factor (`V` below
    /// the diagonal), `tau`, and the block reflector's `T`.
    type Factor<T> = (Vec<T>, Vec<T>, Vec<T>);

    /// The `householder_apply` fixture of one batch entry: colmat's QR reflectors of `a0`
    /// (`backend::fork::householder_reflectors`, LAPACK's `tau` convention) and the block
    /// reflector's triangular factor `T` formed from them by `?larft`. Outside every clock.
    fn householder_fixture<T: LapackScalar + colmat::traits::ComplexField>(
        a0: &[T],
        m: usize,
        n: usize,
        order: Order,
    ) -> Result<Factor<T>, String> {
        let size = m.min(n);
        let (a, tau) = crate::backend::fork::householder_reflectors(a0, m, n, order);
        let mut t = vec![T::zero(); size * size];
        let info = T::larft(
            driver_layout(order),
            m,
            size,
            &a,
            leading(m, n, order),
            &tau,
            &mut t,
            size as i32,
        );
        if info != 0 {
            return Err(format!("larft returned {info}"));
        }
        Ok((a, tau, t))
    }

    /// `householder_apply` in the two public forms the faer rows time, on the same fixture: the
    /// `size = min(m, n)` reflectors of the QR factorization of the seeded `m x n` matrix, applied
    /// on the left to the seeded `m x n` target, in place. Block form: `?larfb` with
    /// `H = I - V T V^H`. Sequence form: `?ormqr`/`?unmqr`. Both compute `Q C`.
    ///
    /// The reflectors are colmat's own ([`householder_fixture`]), handed to the vendor in LAPACK's
    /// convention, because a complex LAPACK QR would produce a different `Q` (its `R` has a real
    /// diagonal, colmat's keeps each pivot's phase) and the two rows would apply different
    /// transformations. The fixture is built once at op construction -- the timed calls only read
    /// it -- as the faer rows build theirs with `qr_in_place`, also outside the clock. As on the
    /// faer rows, the check compares the timed result with the other public form applied to the
    /// same fixture.
    fn householder_op<T: LapackScalar + colmat::traits::ComplexField>(
        m: usize,
        n: usize,
        batch: usize,
        order: Order,
        sequence: bool,
    ) -> Result<Op, String> {
        let size = m.min(n);
        let target = n;
        let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 41));
        let c0 = fill_seeded::<T>(m * target * batch, seed(m, target, 43));
        let work = vec![T::zero(); T::ormqr_scratch(m, target, size).max(1)];
        let factors = (0..batch)
            .map(|bi| householder_fixture(&a0[bi * m * n..(bi + 1) * m * n], m, n, order))
            .collect::<Result<Vec<_>, String>>()?;
        #[derive(Clone, Copy)]
        struct Dims {
            m: usize,
            n: usize,
            size: usize,
            target: usize,
            order: Order,
        }
        struct S<T> {
            factors: Vec<Factor<T>>,
            c0: Vec<T>,
            c: Vec<T>,
            work: Vec<T>,
            block_work: Vec<T>,
            dims: Dims,
            batch: usize,
        }
        // `?larfb`'s workspace is `ldwork x k` with `ldwork = ncols` on the left side.
        let state = S {
            factors,
            c: c0.clone(),
            c0,
            work,
            block_work: vec![T::zero(); target.max(1) * size.max(1)],
            dims: Dims {
                m,
                n,
                size,
                target,
                order,
            },
            batch,
        };
        /// One public form on one fixture, into `c`.
        fn apply<T: LapackScalar>(
            (a, tau, t): &Factor<T>,
            d: Dims,
            work: &mut [T],
            block_work: &mut [T],
            c: &mut [T],
            sequence: bool,
        ) -> Result<(), String> {
            let info = if sequence {
                T::ormqr(
                    driver_layout(d.order),
                    d.m,
                    d.target,
                    d.size,
                    a,
                    leading(d.m, d.n, d.order),
                    tau,
                    c,
                    leading(d.m, d.target, d.order),
                    work,
                )
            } else {
                T::larfb(
                    driver_layout(d.order),
                    d.m,
                    d.target,
                    d.size,
                    a,
                    leading(d.m, d.n, d.order),
                    t,
                    d.size as i32,
                    c,
                    leading(d.m, d.target, d.order),
                    block_work,
                )
            };
            if info != 0 {
                return Err(format!(
                    "{} returned {info}",
                    if sequence { "ormqr" } else { "larfb" }
                ));
            }
            Ok(())
        }
        Ok(op_from(
            state,
            format!(
                "{}(basis={size},target={target})",
                if sequence { "sequence" } else { "block" }
            ),
            // both forms overwrite the target, so prepare restores it (outside the clock)
            |s| {
                s.c.copy_from_slice(&s.c0);
                Ok(())
            },
            move |s, _tk| {
                let len = s.dims.m * s.dims.target;
                for bi in 0..s.batch {
                    apply(
                        &s.factors[bi],
                        s.dims,
                        &mut s.work,
                        &mut s.block_work,
                        &mut s.c[bi * len..(bi + 1) * len],
                        sequence,
                    )?;
                }
                Ok(())
            },
            move |s| {
                let d = s.dims;
                let len = d.m * d.target;
                for bi in 0..s.batch {
                    let mut reference = s.c0[bi * len..(bi + 1) * len].to_vec();
                    apply(
                        &s.factors[bi],
                        d,
                        &mut s.work,
                        &mut s.block_work,
                        &mut reference,
                        !sequence,
                    )?;
                    within::<T>(
                        max_diff(&s.c[bi * len..(bi + 1) * len], &reference),
                        frob(&reference),
                        d.target.max(d.m),
                        "householder_apply",
                    )?;
                }
                Ok(())
            },
        ))
    }

    fn replicate<T: Copy>(values: &[T], batch: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(values.len() * batch);
        for _ in 0..batch {
            out.extend_from_slice(values);
        }
        out
    }

    fn diag_dominant<T: Scalar>(n: usize, batch: usize, order: Order, seed: u64) -> Vec<T> {
        let mut out = fill_seeded::<T>(n * n, seed);
        for i in 0..n {
            out[idx(i, i, n, n, order)].add_assign(T::from_f64(n as f64));
        }
        replicate(&out, batch)
    }

    fn spd<T: Scalar>(n: usize, batch: usize, order: Order, seed: u64) -> Vec<T> {
        let m = fill_seeded::<T>(n * n, seed);
        let mut mm = vec![T::zero(); n * n];
        mul_adj_b(n, n, n, order, &m, &m, &mut mm);
        for i in 0..n {
            mm[idx(i, i, n, n, order)].add_assign(T::from_f64(n as f64 + 1.0));
        }
        replicate(&mm, batch)
    }

    /// **A vendor route must compute the colmat row's quantity**, not merely pass its own
    /// residual check: a call that solves a different problem is worse than `skipped`. Each new
    /// route is run on the exact input its op builds (same generator, same seed) and compared with
    /// the colmat library's own result for that input (`backend::fork::reference`, the calls the
    /// colmat rows time), within the harness tolerance.
    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::backend::fork::reference;
        use colmat::traits::ComplexField;

        fn agree<T: Scalar>(got: &[T], want: &[T], size: usize, what: &str) {
            within::<T>(max_diff(got, want), frob(want), size, what)
                .unwrap_or_else(|e| panic!("{what}: vendor differs from colmat: {e}"));
        }

        fn triangular_solve_agrees<T: LapackBlas + ComplexField>() {
            for order in [Order::F, Order::C] {
                for n in [1, 7, 64, 200] {
                    let u = upper_triangular::<T>(n, 1, order, seed(n, n, 19));
                    let b0 = fill_seeded::<T>(n, seed(n, 1, 23));
                    let mut vendor = b0.clone();
                    T::trsm(
                        layout(order),
                        n,
                        1,
                        &u,
                        leading(n, n, order),
                        &mut vendor,
                        leading(n, 1, order),
                    );
                    let mut colmat = b0;
                    reference::triangular_solve(&u, &mut colmat, n, order);
                    agree(
                        &vendor,
                        &colmat,
                        n,
                        &format!("triangular_solve n={n} {}", order.label()),
                    );
                }
            }
        }

        /// `(ln|det|, det/|det|)` from a packed complete-pivoting factor and its permutations:
        /// `det A = sgn(P) sgn(Q) prod U_ii`. It does not depend on which pivots were chosen.
        fn log_det<T: Scalar>(lu: &[T], rows: &[usize], cols: &[usize], n: usize) -> (f64, T) {
            fn sign(perm: &[usize]) -> f64 {
                let mut seen = vec![false; perm.len()];
                let mut sign = 1.0;
                for start in 0..perm.len() {
                    let (mut i, mut len) = (start, 0);
                    while !seen[i] {
                        seen[i] = true;
                        i = perm[i];
                        len += 1;
                    }
                    if len > 0 && len % 2 == 0 {
                        sign = -sign;
                    }
                }
                sign
            }
            let mut log = 0.0;
            let mut phase = T::from_f64(sign(rows) * sign(cols));
            for i in 0..n {
                let d = lu[idx(i, i, n, n, Order::F)];
                log += d.abs().ln();
                phase = phase.mul(d.mul(T::from_f64(1.0 / d.abs())));
            }
            (log, phase)
        }

        /// The first pivot: the entry of largest `metric` in the whole matrix.
        fn first_pivot<T: Scalar>(a: &[T], n: usize, metric: impl Fn(T) -> f64) -> (usize, usize) {
            let mut best = (0, 0, -1.0);
            for j in 0..n {
                for i in 0..n {
                    let v = metric(a[idx(i, j, n, n, Order::F)]);
                    if v > best.2 {
                        best = (i, j, v);
                    }
                }
            }
            (best.0, best.1)
        }

        /// Real dtypes: complete pivoting picks the entry of largest modulus at every step in
        /// both libraries, so the pivots and the packed factors are the same. Complex dtypes: the
        /// vendor picks the largest modulus `|z|` (`?zgetc2`) and colmat the largest `|re| + |im|`
        /// (`full_pivoting/factor.rs`, `abs1`), a recorded route difference, so the pivot orders
        /// differ; both are complete-pivoting LUs of the same matrix, which is checked through the
        /// determinant they imply (independent of the pivots) and through each side's own first
        /// pivot rule.
        fn full_piv_lu_agrees<T: LapackScalar + ComplexField>() {
            let order = Order::F;
            for n in [1, 8, 64, 150] {
                let a0 = fill_seeded::<T>(n * n, seed(n, n, 7));
                let mut vendor = a0.clone();
                let (mut ipiv, mut jpiv) = (vec![0i32; n], vec![0i32; n]);
                let info = T::getc2(COL_MAJOR, n, &mut vendor, n as i32, &mut ipiv, &mut jpiv);
                assert_eq!(info, 0, "getc2 n={n}");
                let (vrows, vcols) = (
                    interchanges_to_perm(&ipiv).unwrap(),
                    interchanges_to_perm(&jpiv).unwrap(),
                );
                let mut colmat = a0.clone();
                let (rows, cols) = reference::full_piv_lu(&mut colmat, n, order);
                let what = format!("full_piv_lu {n}x{n}");
                if !T::IS_COMPLEX {
                    assert_eq!(vrows, rows, "{what}: row pivots");
                    assert_eq!(vcols, cols, "{what}: column pivots");
                    agree(&vendor, &colmat, n, &format!("{what} packed L\\U"));
                }
                let first = first_pivot(&a0, n, |z: T| z.abs());
                assert_eq!(
                    (vrows[0], vcols[0]),
                    first,
                    "{what}: the vendor pivots on the modulus"
                );
                let (vlog, vphase) = log_det(&vendor, &vrows, &vcols, n);
                let (clog, cphase) = log_det(&colmat, &rows, &cols, n);
                let tol = tol_of::<T>(n);
                assert!(
                    (vlog - clog).abs() <= tol * clog.abs().max(1.0),
                    "{what}: ln|det| {vlog} vs {clog}"
                );
                assert!(
                    max_diff(&[vphase], &[cphase]) <= tol,
                    "{what}: det phase differs"
                );
            }
        }

        fn householder_apply_agrees<T: LapackScalar + ComplexField>() {
            let order = Order::F;
            for (m, n) in [(64, 64), (100, 40), (40, 100), (200, 200)] {
                let size = m.min(n);
                let a0 = fill_seeded::<T>(m * n, seed(m, n, 41));
                let c0 = fill_seeded::<T>(m * n, seed(m, n, 43));
                let mut colmat = c0.clone();
                reference::householder_apply(&a0, &mut colmat, m, n, order);
                let (op_a, tau, t) = householder_fixture(&a0, m, n, order).unwrap();
                let mut work = vec![T::zero(); T::ormqr_scratch(m, n, size).max(1)];
                let mut block = c0.clone();
                let mut block_work = vec![T::zero(); n * size];
                assert_eq!(
                    T::larfb(
                        COL_MAJOR,
                        m,
                        n,
                        size,
                        &op_a,
                        m as i32,
                        &t,
                        size as i32,
                        &mut block,
                        m as i32,
                        &mut block_work
                    ),
                    0
                );
                let mut sequence = c0;
                assert_eq!(
                    T::ormqr(
                        COL_MAJOR,
                        m,
                        n,
                        size,
                        &op_a,
                        m as i32,
                        &tau,
                        &mut sequence,
                        m as i32,
                        &mut work
                    ),
                    0
                );
                agree(
                    &block,
                    &colmat,
                    m.max(n),
                    &format!("householder_apply {m}x{n} block (larfb)"),
                );
                agree(
                    &sequence,
                    &colmat,
                    m.max(n),
                    &format!("householder_apply {m}x{n} sequence (ormqr)"),
                );
            }
        }

        #[test]
        fn vendor_routes_agree_with_colmat() {
            crate::vendor::set_threads(1).unwrap();
            triangular_solve_agrees::<f64>();
            triangular_solve_agrees::<Complex64>();
            triangular_solve_agrees::<f32>();
            triangular_solve_agrees::<num_complex::Complex32>();
            full_piv_lu_agrees::<f64>();
            full_piv_lu_agrees::<Complex64>();
            householder_apply_agrees::<f64>();
            householder_apply_agrees::<Complex64>();
        }

        #[test]
        fn interchanges_compose_in_order() {
            // swap 0<->2, then 1<->2: [0,1,2] -> [2,1,0] -> [2,0,1]
            assert_eq!(interchanges_to_perm(&[3, 3, 3]).unwrap(), vec![2, 0, 1]);
            assert!(interchanges_to_perm(&[0, 2]).is_err());
            assert!(interchanges_to_perm(&[3, 2]).is_err());
        }
    }
}

#[cfg(any(feature = "link-mkl", feature = "link-openblas", feature = "link-aocl"))]
pub use imp::{ops, row_name, ROUTE_DIFFERENCES};

/// No vendor is linked, so no vendor route differs from anything.
#[cfg(not(any(feature = "link-mkl", feature = "link-openblas", feature = "link-aocl")))]
pub const ROUTE_DIFFERENCES: &[(&str, &str)] = &[];

#[cfg(not(any(feature = "link-mkl", feature = "link-openblas", feature = "link-aocl")))]
pub fn row_name() -> &'static str {
    "lapack-none"
}

#[cfg(not(any(feature = "link-mkl", feature = "link-openblas", feature = "link-aocl")))]
pub fn ops(
    _dtype: crate::shared::Dtype,
    _family: &str,
    _case: crate::shared::Case,
    _order: crate::shared::Order,
) -> Result<Vec<crate::op::Op>, String> {
    Err("no vendor library was linked into this binary".to_owned())
}
