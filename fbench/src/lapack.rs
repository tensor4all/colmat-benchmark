//! The vendor BLAS/LAPACK arm: CBLAS `?gemm` plus LAPACKE drivers.
//!
//! The harness declares its own `extern "C"` bindings rather than adding a
//! binding crate: `blas-src`/`lapack-src` (pulled by the `link-*` features in
//! `Cargo.toml`) only emit the native link flags, and the campaign pins the
//! symbol set it calls. Every call goes through LAPACKE's row-major/col-major
//! layout argument, so the `order` axis is a native route, not a transpose the
//! harness performs.
//!
//! The complex ABI is the one flat 16-byte `{re, im}` struct MKL documents as
//! `lapack_complex_double`/`MKL_Complex16`; `num_complex::Complex64` is that
//! layout. (An OpenBLAS built with C99 `double complex` would differ, which is
//! why the MKL arm is the one the campaign runs.)
//!
//! Families with no LAPACK route the harness measures (`triangular_solve`,
//! `full_piv_lu`, `householder_apply`, `upgrade_factor`, `eig`) return an error
//! and are recorded as `skipped`, never as a number.

#[cfg(any(feature = "link-mkl", feature = "link-openblas"))]
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
    const CBLAS_NO_TRANS: i32 = 111;

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
        fn LAPACKE_dgeqrf(layout: i32, m: i32, n: i32, a: *mut f64, lda: i32, tau: *mut f64)
            -> i32;
        fn LAPACKE_zgeqrf(
            layout: i32,
            m: i32,
            n: i32,
            a: *mut Complex64,
            lda: i32,
            tau: *mut Complex64,
        ) -> i32;
        fn LAPACKE_dorgqr(
            layout: i32,
            m: i32,
            n: i32,
            k: i32,
            a: *mut f64,
            lda: i32,
            tau: *const f64,
        ) -> i32;
        fn LAPACKE_zungqr(
            layout: i32,
            m: i32,
            n: i32,
            k: i32,
            a: *mut Complex64,
            lda: i32,
            tau: *const Complex64,
        ) -> i32;
        fn LAPACKE_dgetrf(
            layout: i32,
            m: i32,
            n: i32,
            a: *mut f64,
            lda: i32,
            ipiv: *mut i32,
        ) -> i32;
        fn LAPACKE_zgetrf(
            layout: i32,
            m: i32,
            n: i32,
            a: *mut Complex64,
            lda: i32,
            ipiv: *mut i32,
        ) -> i32;
        fn LAPACKE_dgetrs(
            layout: i32,
            trans: c_char,
            n: i32,
            nrhs: i32,
            a: *const f64,
            lda: i32,
            ipiv: *const i32,
            b: *mut f64,
            ldb: i32,
        ) -> i32;
        fn LAPACKE_zgetrs(
            layout: i32,
            trans: c_char,
            n: i32,
            nrhs: i32,
            a: *const Complex64,
            lda: i32,
            ipiv: *const i32,
            b: *mut Complex64,
            ldb: i32,
        ) -> i32;
        fn LAPACKE_dpotrf(layout: i32, uplo: c_char, n: i32, a: *mut f64, lda: i32) -> i32;
        fn LAPACKE_zpotrf(layout: i32, uplo: c_char, n: i32, a: *mut Complex64, lda: i32) -> i32;
        fn LAPACKE_dgesdd(
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
        ) -> i32;
        fn LAPACKE_zgesdd(
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
        ) -> i32;
        fn LAPACKE_dsyevd(
            layout: i32,
            jobz: c_char,
            uplo: c_char,
            n: i32,
            a: *mut f64,
            lda: i32,
            w: *mut f64,
        ) -> i32;
        fn LAPACKE_zheevd(
            layout: i32,
            jobz: c_char,
            uplo: c_char,
            n: i32,
            a: *mut Complex64,
            lda: i32,
            w: *mut f64,
        ) -> i32;
        fn LAPACKE_dgesv(
            layout: i32,
            n: i32,
            nrhs: i32,
            a: *mut f64,
            lda: i32,
            ipiv: *mut i32,
            b: *mut f64,
            ldb: i32,
        ) -> i32;
        fn LAPACKE_zgesv(
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

    /// The one vendor call every declared dtype has a route for. It is a capability of
    /// its own because `cblas_?gemm` exists for all four dtypes while the LAPACK
    /// drivers below exist, in this harness, for `f64`/`c64` only: a dtype that has the
    /// former must not be made to carry dummy implementations of the latter.
    trait LapackGemm: Scalar {
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
    }

    /// The vendor's LAPACK drivers.
    trait LapackScalar: LapackGemm {
        fn geqrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [Self],
            lda: i32,
            tau: &mut [Self],
        ) -> i32;
        fn orgqr(
            layout: i32,
            m: usize,
            n: usize,
            k: usize,
            a: &mut [Self],
            lda: i32,
            tau: &[Self],
        ) -> i32;
        fn getrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
        ) -> i32;
        fn getrs(
            layout: i32,
            n: usize,
            a: &[Self],
            lda: i32,
            ipiv: &[i32],
            b: &mut [Self],
            ldb: i32,
        ) -> i32;
        fn potrf(layout: i32, n: usize, a: &mut [Self], lda: i32) -> i32;
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
        ) -> i32;
        fn evd(layout: i32, jobz: c_char, n: usize, a: &mut [Self], lda: i32, w: &mut [f64])
            -> i32;
        fn gesv(
            layout: i32,
            n: usize,
            a: &mut [Self],
            lda: i32,
            ipiv: &mut [i32],
            b: &mut [Self],
            ldb: i32,
        ) -> i32;
    }

    impl LapackGemm for f64 {
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
        fn geqrf(layout: i32, m: usize, n: usize, a: &mut [f64], lda: i32, tau: &mut [f64]) -> i32 {
            unsafe {
                LAPACKE_dgeqrf(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_mut_ptr(),
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
        ) -> i32 {
            unsafe {
                LAPACKE_dorgqr(
                    layout,
                    m as i32,
                    n as i32,
                    k as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_ptr(),
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
                LAPACKE_dgetrf(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    ipiv.as_mut_ptr(),
                )
            }
        }
        fn getrs(
            layout: i32,
            n: usize,
            a: &[f64],
            lda: i32,
            ipiv: &[i32],
            b: &mut [f64],
            ldb: i32,
        ) -> i32 {
            unsafe {
                LAPACKE_dgetrs(
                    layout,
                    JOB_N,
                    n as i32,
                    1,
                    a.as_ptr(),
                    lda,
                    ipiv.as_ptr(),
                    b.as_mut_ptr(),
                    ldb,
                )
            }
        }
        fn potrf(layout: i32, n: usize, a: &mut [f64], lda: i32) -> i32 {
            unsafe { LAPACKE_dpotrf(layout, UPLO_L, n as i32, a.as_mut_ptr(), lda) }
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
        ) -> i32 {
            unsafe {
                LAPACKE_dgesdd(
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
                )
            }
        }
        fn evd(layout: i32, jobz: c_char, n: usize, a: &mut [f64], lda: i32, w: &mut [f64]) -> i32 {
            unsafe {
                LAPACKE_dsyevd(
                    layout,
                    jobz,
                    UPLO_L,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    w.as_mut_ptr(),
                )
            }
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
                LAPACKE_dgesv(
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
    }

    impl LapackGemm for Complex64 {
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

    impl LapackGemm for f32 {
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

    impl LapackGemm for num_complex::Complex32 {
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

    impl LapackScalar for Complex64 {
        fn geqrf(
            layout: i32,
            m: usize,
            n: usize,
            a: &mut [Complex64],
            lda: i32,
            tau: &mut [Complex64],
        ) -> i32 {
            unsafe {
                LAPACKE_zgeqrf(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_mut_ptr(),
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
        ) -> i32 {
            unsafe {
                LAPACKE_zungqr(
                    layout,
                    m as i32,
                    n as i32,
                    k as i32,
                    a.as_mut_ptr(),
                    lda,
                    tau.as_ptr(),
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
                LAPACKE_zgetrf(
                    layout,
                    m as i32,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    ipiv.as_mut_ptr(),
                )
            }
        }
        fn getrs(
            layout: i32,
            n: usize,
            a: &[Complex64],
            lda: i32,
            ipiv: &[i32],
            b: &mut [Complex64],
            ldb: i32,
        ) -> i32 {
            unsafe {
                LAPACKE_zgetrs(
                    layout,
                    JOB_N,
                    n as i32,
                    1,
                    a.as_ptr(),
                    lda,
                    ipiv.as_ptr(),
                    b.as_mut_ptr(),
                    ldb,
                )
            }
        }
        fn potrf(layout: i32, n: usize, a: &mut [Complex64], lda: i32) -> i32 {
            unsafe { LAPACKE_zpotrf(layout, UPLO_L, n as i32, a.as_mut_ptr(), lda) }
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
        ) -> i32 {
            unsafe {
                LAPACKE_zgesdd(
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
        ) -> i32 {
            unsafe {
                LAPACKE_zheevd(
                    layout,
                    jobz,
                    UPLO_L,
                    n as i32,
                    a.as_mut_ptr(),
                    lda,
                    w.as_mut_ptr(),
                )
            }
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
                LAPACKE_zgesv(
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
    }

    /// The row name the CSV carries: derived from what was linked, not requested.
    pub fn row_name() -> &'static str {
        if cfg!(feature = "link-mkl") {
            "lapack-mkl"
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

    /// The leading dimension of an `m x n` buffer stored in `order`.
    fn leading(m: usize, n: usize, order: Order) -> i32 {
        match order {
            Order::C => n as i32,
            Order::F => m as i32,
        }
    }

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

    /// The vendor rows for the two `f32` dtypes: `?gemm` exists for both, the LAPACK
    /// drivers do not (in this harness), so every other family is a skip with the reason
    /// in it rather than a number or a dummy driver.
    fn ops_gemm_only<T: LapackGemm>(
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
            Dtype::F32 => ops_gemm_only::<f32>(dtype, family, case, order),
            Dtype::C32 => ops_gemm_only::<num_complex::Complex32>(dtype, family, case, order),
        }
    }

    fn ops_typed<T: LapackScalar>(
        family: &str,
        case: Case,
        order: Order,
    ) -> Result<Vec<Op>, String> {
        let m = case.m;
        let n = case.n;
        let batch = case.batch;
        match family {
            "gemm" => Ok(vec![gemm_op::<T>(m, case.k.unwrap_or(m), n, batch, order)?]),
            "qr_factor" => Ok(vec![qr_op::<T>(m, n, batch, order, false)?]),
            "qr" => Ok(vec![qr_op::<T>(m, n, batch, order, true)?]),
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
            other => Err(format!(
                "{other} has no LAPACK route in this harness; the vendor row is skipped"
            )),
        }
    }

    fn gemm_op<T: LapackGemm>(
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

    /// `qr_factor` times `geqrf`; `qr` also forms Q with `orgqr`. The check
    /// reconstructs `A = Q R` from the retained factors (and Q when requested).
    fn qr_op<T: LapackScalar>(
        m: usize,
        n: usize,
        batch: usize,
        order: Order,
        with_q: bool,
    ) -> Result<Op, String> {
        let salt = if with_q { 3 } else { m.min(n) };
        let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, salt));
        struct S<T> {
            a0: Vec<T>,
            a: Vec<Vec<T>>,
            tau: Vec<Vec<T>>,
            r: Vec<Vec<T>>,
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
            with_q: bool,
        }
        // Every buffer the vendor call writes is the harness's, so it is allocated here, once,
        // and reused by every call (docs/reliable-measurement.md, "Allocation and the clock").
        // `r` stays zero below its diagonal: only the upper trapezoid is ever written.
        let state = S {
            a: a0.chunks_exact(m * n).map(|a| a.to_vec()).collect(),
            tau: vec![vec![T::zero(); m.min(n)]; batch],
            r: if with_q {
                vec![vec![T::zero(); m.min(n) * n]; batch]
            } else {
                Vec::new()
            },
            a0,
            m,
            n,
            batch,
            order,
            with_q,
        };
        Ok(op_from(
            state,
            String::new(),
            // The buffers are reused, so prepare spoils the retained result (outside the clock) and a
            // check can only pass on what the next timed call writes.
            |s| {
                if s.with_q {
                    for r in &mut s.r {
                        r[0] = T::from_f64(1e30);
                    }
                } else {
                    for bi in 0..s.batch {
                        s.a[bi].copy_from_slice(&s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n]);
                        s.tau[bi].fill(T::zero());
                    }
                }
                Ok(())
            },
            |s, _tk| {
                let r = s.m.min(s.n);
                for bi in 0..s.batch {
                    if s.with_q {
                        s.a[bi].copy_from_slice(&s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n]);
                    }
                    let a = &mut s.a[bi];
                    let tau = &mut s.tau[bi];
                    let info = T::geqrf(
                        layout(s.order),
                        s.m,
                        s.n,
                        a,
                        leading(s.m, s.n, s.order),
                        tau,
                    );
                    if info != 0 {
                        return Err(format!("geqrf returned {info}"));
                    }
                    if s.with_q {
                        let rmat = &mut s.r[bi];
                        for i in 0..r {
                            for j in i..s.n {
                                rmat[idx(i, j, r, s.n, s.order)] = a[idx(i, j, s.m, s.n, s.order)];
                            }
                        }
                        // Q is m x r with r = min(m, n) columns; asking orgqr for
                        // more columns than Q has (n > m on a wide shape) is invalid
                        // and MKL rejects it. The m x n buffer keeps its own leading
                        // dimension, so only the requested column count changes.
                        let info = T::orgqr(
                            layout(s.order),
                            s.m,
                            r,
                            r,
                            a,
                            leading(s.m, s.n, s.order),
                            tau,
                        );
                        if info != 0 {
                            return Err(format!("orgqr returned {info}"));
                        }
                    }
                }
                Ok(())
            },
            |s| {
                let r = s.m.min(s.n);
                for bi in 0..s.batch {
                    let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                    let mut a = s.a[bi].clone();
                    let mut rmat = if s.with_q {
                        s.r[bi].clone()
                    } else {
                        vec![T::zero(); r * s.n]
                    };
                    if !s.with_q {
                        // Derive Q from the ACTUAL compact factors, never re-factor input.
                        for i in 0..r {
                            for j in i..s.n {
                                rmat[idx(i, j, r, s.n, s.order)] = a[idx(i, j, s.m, s.n, s.order)];
                            }
                        }
                        let info = T::orgqr(
                            layout(s.order),
                            s.m,
                            r,
                            r,
                            &mut a,
                            leading(s.m, s.n, s.order),
                            &s.tau[bi],
                        );
                        if info != 0 {
                            return Err(format!("orgqr returned {info}"));
                        }
                    }
                    let mut qmat = vec![T::zero(); s.m * r];
                    for i in 0..s.m {
                        for j in 0..r {
                            qmat[idx(i, j, s.m, r, s.order)] = a[idx(i, j, s.m, s.n, s.order)];
                        }
                    }
                    let mut recon = vec![T::zero(); s.m * s.n];
                    mul(s.m, r, s.n, s.order, &qmat, &rmat, &mut recon);
                    within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "qr")?;
                    within::<T>(gram_residual(&qmat, s.m, r, s.order), 1.0, s.m, "qr Q")?;
                }
                Ok(())
            },
        ))
    }

    /// `lu` times `getrf`; the check rebuilds `L` and `U` from the factors and
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
            outputs: (0..batch).map(|_| (vec![T::zero(); m * n], vec![0i32; size])).collect(),
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
                        layout(s.order),
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
                    let info = T::potrf(layout(s.order), s.n, a, leading(s.n, s.n, s.order));
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
            work: (0..batch).map(|_| (vec![T::zero(); n * n], vec![0i32; n])).collect(),
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
                        layout(s.order),
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

    /// `svd_values` times `gesdd` with `jobz='N'`; `svd_thin` with `jobz='S'`.
    fn svd_op<T: LapackScalar>(
        m: usize,
        n: usize,
        batch: usize,
        order: Order,
        thin: bool,
    ) -> Result<Op, String> {
        let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, if thin { 31 } else { 29 }));
        struct S<T> {
            a0: Vec<T>,
            work: Vec<Vec<T>>,
            outputs: Vec<(Vec<f64>, Vec<T>, Vec<T>)>,
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
            // The buffers are reused, so prepare spoils the retained result (outside the clock) and a
            // check can only pass on what the next timed call writes.
            |s| {
                for (sigma, u, _) in &mut s.outputs {
                    sigma[0] = -1.0;
                    if let Some(x) = u.first_mut() {
                        *x = T::from_f64(1e30);
                    }
                }
                Ok(())
            },
            |s, _tk| {
                let r = s.m.min(s.n);
                for bi in 0..s.batch {
                    let a = &mut s.work[bi];
                    a.copy_from_slice(&s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n]);
                    let (sigma, u, vt) = &mut s.outputs[bi];
                    let mut dummy_u = [T::zero()];
                    let mut dummy_vt = [T::zero()];
                    let jobz = if s.thin { JOB_S } else { JOB_N };
                    let info = T::gesdd(
                        layout(s.order),
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
        struct S<T> {
            a0: Vec<T>,
            outputs: Vec<(Vec<T>, Vec<f64>)>,
            n: usize,
            batch: usize,
            order: Order,
        }
        let state = S {
            outputs: (0..batch).map(|_| (vec![T::zero(); n * n], vec![0.0f64; n])).collect(),
            a0,
            n,
            batch,
            order,
        };
        Ok(op_from(
            state,
            String::new(),
            // The buffers are reused, so prepare spoils the retained result (outside the clock) and a
            // check can only pass on what the next timed call writes.
            |s| {
                for (a, _) in &mut s.outputs {
                    a[0] = T::from_f64(1e30);
                }
                Ok(())
            },
            |s, _tk| {
                for bi in 0..s.batch {
                    let (a, w) = &mut s.outputs[bi];
                    a.copy_from_slice(&s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n]);
                    let info = T::evd(
                        layout(s.order),
                        JOB_V,
                        s.n,
                        a,
                        leading(s.n, s.n, s.order),
                        w,
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
                    within::<T>(gram_residual(&u, s.n, s.n, s.order), 1.0, s.n, "eigh U")?;
                    let ws = w.iter().map(|x| T::from_f64(*x)).collect::<Vec<_>>();
                    let mut au = vec![T::zero(); s.n * s.n];
                    mul(s.n, s.n, s.n, s.order, a0, &u, &mut au);
                    let us = scale_cols(&u, s.n, s.n, &ws, s.order);
                    within::<T>(max_diff(&au, &us), scale, s.n, "eigh")?;
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
}

#[cfg(any(feature = "link-mkl", feature = "link-openblas"))]
pub use imp::{ops, row_name};

#[cfg(not(any(feature = "link-mkl", feature = "link-openblas")))]
pub fn row_name() -> &'static str {
    "lapack-none"
}

#[cfg(not(any(feature = "link-mkl", feature = "link-openblas")))]
pub fn ops(
    _dtype: crate::shared::Dtype,
    _family: &str,
    _case: crate::shared::Case,
    _order: crate::shared::Order,
) -> Result<Vec<crate::op::Op>, String> {
    Err("no vendor library was linked into this binary".to_owned())
}
