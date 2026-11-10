//! The per-crate adapter.
//!
//! `faer_fork` (`t4a-faer`) and `faer_upstream` (`faer`) are distinct dependency
//! keys with distinct `Mat`, `Par` and trait types, so the glue that turns a plain
//! `&[T]` and a shape into that crate's calls is written twice. `define_backend!`
//! expands the same body once per crate; the family logic and every correctness
//! check it drives are written once, in the macro body and in `shared`.
//!
//! Each family returns an [`Op`]: `prepare` restores the caller-visible inputs,
//! `run` is the measured call, and `check` is a residual or norm identity computed
//! from that provider's own result against provider-independent reference math.
//! A family that is not implemented returns `Err(note)`; the caller records it as

/// How a row runs its parallel regions, as a plain value. The fork's `Par` is not `Copy` from 0.25
/// on (a team is exclusive authority), so an op keeps this and builds a `Par` for every call; the
/// same value serves the baseline, where `Par` is `Copy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParCfg {
    Seq,
    Rayon(usize),
}

/// `ParCfg` into the fork's `Par` (for running) and into what its workspace queries take.
///
/// The fork is linked at two API generations and the library is not edited to bridge them:
///
/// * baseline (0.24.x): `Par` is `Copy`, the `*_scratch` functions take a `Par`, and there is a
///   process-wide parallelism setting (which the fork rows do **not** use);
/// * 0.25 candidate: `Par<'team>` is not `Copy`, the `*_scratch` functions take a `ParShape`, and
///   the process-wide setting is removed.
///
/// `scripts/build_for_faer_rev.sh` looks at the fork checkout (`faer/src/lib.rs` declares
/// `pub enum ParShape` or not) and turns on the harness feature `fork-par-shape` for the second
/// generation. The row's budget is the harness's own [`FORK_THREADS`], never a library global.
pub mod fork_par {
    use super::ParCfg;
    use faer_fork::Par;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FORK_THREADS: AtomicUsize = AtomicUsize::new(1);

    #[cfg(feature = "fork-par-shape")]
    pub type Run = Par<'static>;
    #[cfg(not(feature = "fork-par-shape"))]
    pub type Run = Par<'static>;
    #[cfg(feature = "fork-par-shape")]
    pub type Shape = faer_fork::ParShape;
    #[cfg(not(feature = "fork-par-shape"))]
    pub type Shape = Par<'static>;

    /// The old generations (before tensor4all/faer#85) build a `Par` from the row's budget; with
    /// the team API there is no such constructor, because a token is exclusive authority over
    /// reserved members and only the timing loop's team hands one out.
    #[cfg(not(feature = "fork-team"))]
    pub fn mk(c: ParCfg) -> Run {
        match c {
            ParCfg::Seq => Par::Seq,
            ParCfg::Rayon(n) => Par::rayon(n),
        }
    }

    /// The parallelism of one measured call, from the row's token.
    ///
    /// With the team API (`fork-team`, detected by `scripts/build_for_faer_rev.sh`) the row's
    /// parallelism is the team the timing loop owns: the token borrows it, so every call reuses the
    /// same reservation and nothing is acquired inside a clocked region. A budget of one thread, or
    /// a row whose op is not the fork, hands out `Par::Seq`. Without the team API the fork's `Par`
    /// is `Copy` and is built from the row's budget per call, as it was before #85.
    #[cfg(feature = "fork-team")]
    pub fn mk_from<'a>(tk: &'a mut crate::op::Token<'_>, _c: ParCfg) -> Par<'a> {
        match tk {
            crate::op::Token::None(_) => Par::Seq,
            crate::op::Token::Fork(par) => par.reborrow(),
        }
    }
    #[cfg(not(feature = "fork-team"))]
    pub fn mk_from(_tk: &mut crate::op::Token<'_>, c: ParCfg) -> Run {
        mk(c)
    }

    /// The parallelism of the *verification* code (`check` closures, reference factorizations): the
    /// clock is not running there, and with the team API a token exists only inside the team the
    /// timing loop owns, so the reconstruction runs on the caller.
    #[cfg(feature = "fork-team")]
    pub fn verify_par(_c: ParCfg) -> Par<'static> {
        Par::Seq
    }
    #[cfg(not(feature = "fork-team"))]
    pub fn verify_par(c: ParCfg) -> Run {
        mk(c)
    }

    /// The shape the workspace queries take. With the team API it is the shape of the team of the
    /// row's budget (`ParShape::team`), which is what the token will be; the sizes only need the
    /// width, not the authority.
    #[cfg(all(feature = "fork-par-shape", feature = "fork-team"))]
    pub fn shape(c: ParCfg) -> Shape {
        match c {
            ParCfg::Seq => faer_fork::ParShape::Seq,
            ParCfg::Rayon(n) => faer_fork::ParShape::team(n),
        }
    }
    #[cfg(all(feature = "fork-par-shape", not(feature = "fork-team")))]
    pub fn shape(c: ParCfg) -> Shape {
        match c {
            ParCfg::Seq => faer_fork::ParShape::Seq,
            ParCfg::Rayon(n) => faer_fork::ParShape::rayon(n),
        }
    }
    #[cfg(not(feature = "fork-par-shape"))]
    pub fn shape(c: ParCfg) -> Shape {
        mk(c)
    }

    pub fn degree() -> usize {
        FORK_THREADS.load(Ordering::Relaxed)
    }

    pub fn set_threads(threads: usize) {
        FORK_THREADS.store(threads.max(1), Ordering::Relaxed);
    }
}

/// `ParCfg` into upstream's `Par` (`Copy`, and also what its workspace queries take). Upstream keeps
/// its own process-wide setting: the high level calls of the upstream row read it, inside the
/// spindle group the row holds (`main.rs` `measure_row`).
pub mod upstream_par {
    use super::ParCfg;
    use faer_upstream::Par;

    #[allow(dead_code)]
    pub type Run = Par;
    #[allow(dead_code)]
    pub type Shape = Par;

    pub fn mk(c: ParCfg) -> Par {
        match c {
            ParCfg::Seq => Par::Seq,
            ParCfg::Rayon(n) => Par::rayon(n),
        }
    }

    /// Upstream's `Par` is `Copy` and comes from its own global setting, which the row sets before
    /// the loop: the token carries nothing for it.
    pub fn mk_from(_tk: &mut crate::op::Token<'_>, c: ParCfg) -> Par {
        mk(c)
    }
    pub fn shape(c: ParCfg) -> Par {
        mk(c)
    }
    pub fn verify_par(c: ParCfg) -> Par {
        mk(c)
    }
    pub fn degree() -> usize {
        faer_upstream::get_global_parallelism().degree()
    }
    pub fn set_threads(threads: usize) {
        let par = if threads <= 1 { Par::Seq } else { Par::rayon(threads) };
        faer_upstream::set_global_parallelism(par);
    }
}


macro_rules! define_backend {
    ($modname:ident, $krate:ident, $parmod:ident, $ops:ident) => {
        pub mod $modname {
            use super::$parmod as parmod;
            use super::ParCfg;
            use crate::op::{op_from, Op};
            use crate::shared::*;
            use num_complex::Complex64;

            use $krate::dyn_stack::{MemBuffer, MemStack, StackReq};
            use $krate::linalg::householder;
            use $krate::linalg::qr::no_pivoting::factor as qrf;
            use $krate::traits::math_utils::from_f64;
            use $krate::traits::ComplexField;
            use $krate as fk;
            use $krate::{Accum, Conj, MatMut, MatRef};

            /// The row's budget: the fork's own atomic, or upstream's global setting.
            pub fn set_threads(threads: usize) {
                parmod::set_threads(threads);
            }

            /// The parallelism the row asked for when an op is built.
            fn cfg_par() -> ParCfg {
                match parmod::degree() {
                    0 | 1 => ParCfg::Seq,
                    n => ParCfg::Rayon(n),
                }
            }

            fn strides(m: usize, n: usize, order: Order) -> (isize, isize) {
                match order {
                    Order::C => (n as isize, 1),
                    Order::F => (1, m as isize),
                }
            }

            fn mref<'a, T: ComplexField>(d: &'a [T], m: usize, n: usize, order: Order) -> MatRef<'a, T> {
                let (rs, cs) = strides(m, n, order);
                // SAFETY: the caller owns `d` for the view's lifetime and the strides
                // describe exactly that buffer.
                unsafe { MatRef::from_raw_parts(d.as_ptr(), m, n, rs, cs) }
            }

            fn mmut<'a, T: ComplexField>(d: &'a mut [T], m: usize, n: usize, order: Order) -> MatMut<'a, T> {
                let (rs, cs) = strides(m, n, order);
                // SAFETY: as `mref`, with an exclusive borrow of the buffer.
                unsafe { MatMut::from_raw_parts_mut(d.as_mut_ptr(), m, n, rs, cs) }
            }

            /// Copy a matrix into a flat buffer in the requested layout.
            fn to_layout<T: ComplexField>(mat: MatRef<'_, T>, m: usize, n: usize, order: Order) -> Vec<T> {
                let mut out = vec![T::zero_impl(); m * n];
                for j in 0..n {
                    for i in 0..m {
                        out[idx(i, j, m, n, order)] = mat[(i, j)].clone();
                    }
                }
                out
            }

            fn diag_vec<T: ComplexField>(diag: $krate::diag::DiagRef<'_, T>) -> Vec<T> {
                (0..diag.dim()).map(|i| diag[i].clone()).collect()
            }

            /// The residual check: `max|residual| <= tol * scale`.
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

            /// Build the ops for one case, per dtype.
            pub fn ops(dtype: Dtype, family: &str, case: Case, order: Order) -> Result<Vec<Op>, String> {
                match dtype {
                    Dtype::F64 => ops_typed::<f64>(family, case, order),
                    Dtype::C64 => ops_typed::<Complex64>(family, case, order),
                    Dtype::F32 => ops_typed::<f32>(family, case, order),
                    Dtype::C32 => ops_typed::<num_complex::Complex32>(family, case, order),
                }
            }

            fn ops_typed<T: Scalar + ComplexField>(
                family: &str,
                case: Case,
                order: Order,
            ) -> Result<Vec<Op>, String>
            where
                T::Real: Scalar,
            {
                let m = case.m;
                let n = case.n;
                let batch = case.batch;
                match family {
                    "gemm" => Ok(vec![gemm_op::<T>(m, case.k.unwrap_or(m), n, batch, order)?]),
                    "qr_factor" => Ok(vec![qr_factor_op::<T>(m, n, batch, order)?]),
                    "qr" => Ok(vec![qr_op::<T>(m, n, batch, order)?]),
                    // The LU factors are defined for a rectangular matrix too:
                    // `dgetrf` takes any shape and faer's `lu_in_place` does as
                    // well, with `min(m, n)` pivots and a trapezoidal `L` or
                    // `U`. So neither is gated to square.
                    "lu" => Ok(vec![lu_op::<T>(m, n, batch, order)?, lu_in_place_op::<T>(m, n, batch, order)?]),
                    "full_piv_lu" => Ok(vec![
                        full_piv_lu_op::<T>(m, n, batch, order)?,
                        full_piv_lu_in_place_op::<T>(m, n, batch, order)?,
                    ]),
                    "cholesky" => {
                        square(case, family)?;
                        Ok(vec![cholesky_op::<T>(m, batch, order)?, cholesky_in_place_op::<T>(m, batch, order)?])
                    }
                    "solve" => {
                        square(case, family)?;
                        Ok(vec![solve_op::<T>(m, batch, order)?, solve_in_place_op::<T>(m, batch, order)?])
                    }
                    "triangular_solve" => {
                        square(case, family)?;
                        Ok(vec![triangular_solve_op::<T>(m, batch, order)?])
                    }
                    "svd_values" => Ok(vec![
                        svd_values_op::<T>(m, n, batch, order)?,
                        svd_in_place_op::<T>(m, n, batch, order, false)?,
                    ]),
                    "svd_thin" => Ok(vec![
                        svd_thin_op::<T>(m, n, batch, order)?,
                        svd_in_place_op::<T>(m, n, batch, order, true)?,
                    ]),
                    "eigh" => {
                        square(case, family)?;
                        Ok(vec![eigh_op::<T>(m, batch, order)?, eigh_in_place_op::<T>(m, batch, order)?])
                    }
                    "householder_apply" => householder_ops::<T>(m, n, batch, order),
                    // Deliberately not written yet. `upgrade_factor` needs a valid
                    // blocked-tau layout built by hand to be measured and checked
                    // honestly, and `eig` is the non-symmetric solver the first
                    // version does not touch. Both are recorded as skipped.
                    other => Err(format!(
                        "{other} is not implemented for the faer arms in this harness"
                    )),
                }
            }

            // ---------------------------------------------------------- gemm

            fn gemm_op<T: Scalar + ComplexField>(
                m: usize,
                k: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let a = fill_seeded::<T>(m * k * batch, seed(m, k, n));
                let b = fill_seeded::<T>(k * n * batch, seed(k, n, m));
                // Built when `check` first needs it, not at construction: the naive O(m*n*k) reference is
                // outside every timed region either way, and a row that is never checked never pays it.
                let c_ref: Vec<T> = Vec::new();
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
                let par = cfg_par();
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
                    move |s, mut tk| {
                        for bi in 0..s.batch {
                            let dst = mmut(&mut s.c[bi * s.m * s.n..], s.m, s.n, s.order);
                            let lhs = mref(&s.a[bi * s.m * s.k..], s.m, s.k, s.order);
                            let rhs = mref(&s.b[bi * s.k * s.n..], s.k, s.n, s.order);
                            $krate::linalg::matmul::matmul(dst, Accum::Replace, lhs, rhs, T::one(), parmod::mk_from(&mut tk, par));
                        }
                        Ok(())
                    },
                    |s| {
                        if s.c_ref.is_empty() {
                            let mut c_ref = vec![T::zero(); s.m * s.n * s.batch];
                            for bi in 0..s.batch {
                                mul(
                                    s.m,
                                    s.k,
                                    s.n,
                                    s.order,
                                    &s.a[bi * s.m * s.k..(bi + 1) * s.m * s.k],
                                    &s.b[bi * s.k * s.n..(bi + 1) * s.k * s.n],
                                    &mut c_ref[bi * s.m * s.n..(bi + 1) * s.m * s.n],
                                );
                            }
                            s.c_ref = c_ref;
                        }
                        let diff = max_diff(&s.c, &s.c_ref);
                        within::<T>(diff, frob(&s.c_ref), s.n.max(s.m), "gemm")
                    },
                ))
            }

            // ------------------------------------------------------- qr_factor

            fn qr_factor_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let size = m.min(n);
                let block_size = qrf::recommended_block_size::<T>(m, n);
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, size));
                let par = cfg_par();
                // The check rebuilds the compact `Q` from the basis and the
                // coefficients the timed call left behind, so it needs the
                // sequence apply's own temporary on top of the factorisation's.
                let scratch = StackReq::all_of(&[
                    qrf::qr_in_place_scratch::<T>(m, n, block_size, parmod::shape(par), Default::default()),
                    householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                        m,
                        size,
                        size,
                    ),
                ]);
                struct S<T> {
                    a: Vec<T>,
                    a0: Vec<T>,
                    qc: Vec<T>,
                    mem: MemBuffer,
                    m: usize,
                    n: usize,
                    block_size: usize,
                    size: usize,
                    batch: usize,
                    order: Order,
                    par: ParCfg,
                }
                let state = S {
                    a: a0.clone(),
                    a0,
                    qc: vec![T::zero(); block_size * size * batch],
                    mem: MemBuffer::new(scratch),
                    m,
                    n,
                    block_size,
                    size,
                    batch,
                    order,
                    par,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| {
                        s.a.copy_from_slice(&s.a0);
                        s.qc.fill(T::zero());
                        Ok(())
                    },
                    |s, mut tk| {
                        for bi in 0..s.batch {
                            let a = mmut(&mut s.a[bi * s.m * s.n..], s.m, s.n, s.order);
                            let qc = mmut(
                                &mut s.qc[bi * s.block_size * s.size..],
                                s.block_size,
                                s.size,
                                Order::F,
                            );
                            qrf::qr_in_place(
                                a,
                                qc,
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem),
                                Default::default(),
                            );
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let a = &s.a[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let qc = &s.qc[bi * s.block_size * s.size
                                ..(bi + 1) * s.block_size * s.size];
                            // Against the state the timed call left, which is what
                            // `Op`'s own contract says a check inspects: `R` is the
                            // upper trapezoid of what is now in the factor buffer and
                            // `Q` is the compact basis applied to an identity, the
                            // same two things `Qr::compute_thin_Q` would build. The
                            // previous check re-ran the *composite* on the original
                            // fixture, which copies into a column-major matrix first,
                            // so it never saw what a row-major call left.
                            let mut r = vec![T::zero(); s.size * s.n];
                            for i in 0..s.size {
                                for j in i..s.n {
                                    r[idx(i, j, s.size, s.n, s.order)] =
                                        a[idx(i, j, s.m, s.n, s.order)];
                                }
                            }
                            let mut q = vec![T::zero(); s.m * s.size];
                            for i in 0..s.size {
                                q[idx(i, i, s.m, s.size, s.order)] = T::one();
                            }
                            householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                                mref(a, s.m, s.n, s.order).get(.., ..s.size),
                                mref(qc, s.block_size, s.size, Order::F),
                                Conj::No,
                                mmut(&mut q, s.m, s.size, s.order),
                                parmod::verify_par(s.par),
                                MemStack::new(&mut s.mem),
                            );
                            let mut recon = vec![T::zero(); s.m * s.n];
                            mul(s.m, s.size, s.n, s.order, &q, &r, &mut recon);
                            within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "qr_factor")?;
                            within::<T>(gram_residual(&q, s.m, s.size, s.order), 1.0, s.m, "qr Q")?;
                        }
                        Ok(())
                    },
                ))
            }


            #[allow(dead_code)]
            /// The singular value checks shared by the families that return values only.
            fn check_singular_values<T: Scalar + ComplexField>(
                a0: &[T],
                sv: &[T::Real],
                m: usize,
                n: usize,
            ) -> Result<(), String>
            where
                T::Real: Scalar,
            {
                let r = sv.len();
                let eps = tol_of::<T>(n.max(m));
                let neg_eps = from_f64::<T::Real>(-eps);
                let eps_r = from_f64::<T::Real>(eps);
                for i in 0..r {
                    let value = <T::Real as Scalar>::abs(sv[i]);
                    if !value.is_finite() || sv[i] < neg_eps {
                        return Err(format!("singular value {i} is negative: {value}"));
                    }
                    if i + 1 < r && sv[i] < sv[i + 1] - eps_r {
                        return Err("singular values are not sorted".to_owned());
                    }
                }
                let sum: f64 = sv
                    .iter()
                    .map(|x| {
                        let v = <T::Real as Scalar>::abs(*x);
                        v * v
                    })
                    .sum();
                let scale = frob(a0);
                within::<T>((sum.sqrt() - scale).abs(), scale, n.max(m), "svd_values norm identity")
            }

            $ops!();

            // -------------------------------------------------- householder_apply

            fn householder_ops<T: Scalar + ComplexField>(
                m: usize, n: usize, batch: usize, order: Order,
            ) -> Result<Vec<Op>, String> {
                Ok(vec![householder_op::<T>(m, n, batch, order, false)?,
                        householder_op::<T>(m, n, batch, order, true)?])
            }

            fn householder_op<T: Scalar + ComplexField>(
                m: usize, n: usize, batch: usize, order: Order, sequence: bool,
            ) -> Result<Op, String> {
                let size = m.min(n);
                let target = n;
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 41));
                let c0 = fill_seeded::<T>(m * target * batch, seed(m, target, 43));
                let par = cfg_par();
                let block_scratch = householder::apply_block_householder_on_the_left_in_place_scratch::<T>(
                    m, size, target,
                );
                let seq_scratch =
                    householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                        m, size, target,
                    );
                let scratch = StackReq::any_of(&[
                    block_scratch,
                    seq_scratch,
                    qrf::qr_in_place_scratch::<T>(m, n, size, parmod::shape(par), Default::default()),
                ]);
                struct S<T> {
                    a: Vec<T>,
                    a0: Vec<T>,
                    qc: Vec<T>,
                    c0: Vec<T>,
                    c: Vec<T>,
                    mem: MemBuffer,
                    m: usize,
                    n: usize,
                    size: usize,
                    target: usize,
                    batch: usize,
                    order: Order,
                    par: ParCfg,
                }
                let state = S {
                    a: a0.clone(),
                    a0,
                    qc: vec![T::zero(); size * size * batch],
                    c0: c0.clone(),
                    c: c0,
                    mem: MemBuffer::new(scratch),
                    m,
                    n,
                    size,
                    target,
                    batch,
                    order,
                    par,
                };
                let block = op_from(
                    state,
                    format!("{}(basis={size},target={target})", if sequence { "sequence" } else { "block" }),
                    |s| {
                        s.a.copy_from_slice(&s.a0);
                        s.qc.fill(T::zero());
                        s.c.copy_from_slice(&s.c0);
                        for bi in 0..s.batch {
                            let a = mmut(&mut s.a[bi * s.m * s.n..], s.m, s.n, s.order);
                            let qc = mmut(&mut s.qc[bi * s.size * s.size..], s.size, s.size, Order::F);
                            qrf::qr_in_place(a, qc, parmod::verify_par(s.par), MemStack::new(&mut s.mem), Default::default());
                        }
                        Ok(())
                    },
                    move |s, mut tk| {
                        for bi in 0..s.batch {
                            let basis = mref(&s.a[bi * s.m * s.n..], s.m, s.n, s.order).get(.., ..s.size);
                            let factor = mref(&s.qc[bi * s.size * s.size..], s.size, s.size, Order::F);
                            let dst = mmut(&mut s.c[bi * s.m * s.target..], s.m, s.target, s.order);
                            if sequence {
                                householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                                    basis, factor, Conj::No, dst, parmod::mk_from(&mut tk, s.par), MemStack::new(&mut s.mem));
                            } else {
                                householder::apply_block_householder_on_the_left_in_place_with_conj(
                                    basis, factor, Conj::No, dst, parmod::mk_from(&mut tk, s.par), MemStack::new(&mut s.mem));
                            }
                        }
                        Ok(())
                    },
                    move |s| {
                        // Compare the actual timed result with the other public form.
                        for bi in 0..s.batch {
                            let mut a = s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n].to_vec();
                            let mut qc = vec![T::zero(); s.size * s.size];
                            qrf::qr_in_place(
                                mmut(&mut a, s.m, s.n, s.order),
                                mmut(&mut qc, s.size, s.size, Order::F),
                                parmod::verify_par(s.par),
                                MemStack::new(&mut s.mem),
                                Default::default(),
                            );
                            let mut reference = s.c0[bi * s.m * s.target..(bi + 1) * s.m * s.target].to_vec();
                            let basis = mref(&a, s.m, s.n, s.order).get(.., ..s.size);
                            let factor = mref(&qc, s.size, s.size, Order::F);
                            let dst = mmut(&mut reference, s.m, s.target, s.order);
                            if sequence {
                                householder::apply_block_householder_on_the_left_in_place_with_conj(
                                    basis, factor, Conj::No, dst, parmod::verify_par(s.par), MemStack::new(&mut s.mem));
                            } else {
                                householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                                    basis, factor, Conj::No, dst, parmod::verify_par(s.par), MemStack::new(&mut s.mem));
                            }
                            let got = &s.c[bi * s.m * s.target..(bi + 1) * s.m * s.target];
                            within::<T>(
                                max_diff(got, &reference),
                                frob(&reference),
                                s.target.max(s.m),
                                "householder_apply",
                            )?;
                        }
                        Ok(())
                    },
                );
                Ok(block)
            }

            fn replicate<T: Copy>(values: &[T], batch: usize) -> Vec<T> {
                let mut out = Vec::with_capacity(values.len() * batch);
                for _ in 0..batch {
                    out.extend_from_slice(values);
                }
                out
            }

            /// `M + n I`: general, well conditioned.
            fn diag_dominant<T: Scalar>(n: usize, batch: usize, order: Order, seed: u64) -> Vec<T> {
                let mut out = fill_seeded::<T>(n * n, seed);
                for i in 0..n {
                    out[idx(i, i, n, n, order)].add_assign(T::from_f64(n as f64));
                }
                replicate(&out, batch)
            }

            /// A diagonally dominant upper triangular matrix.
            fn upper_triangular<T: Scalar>(n: usize, batch: usize, order: Order, seed: u64) -> Vec<T> {
                let raw = fill_seeded::<T>(n * n, seed);
                let mut out = vec![T::zero(); n * n];
                for j in 0..n {
                    for i in 0..n {
                        if i <= j {
                            out[idx(i, j, n, n, order)] = raw[idx(i, j, n, n, order)];
                        }
                    }
                    out[idx(j, j, n, n, order)].add_assign(T::from_f64(n as f64 + 1.0));
                }
                replicate(&out, batch)
            }

            /// `M M^H + n I`: a Hermitian positive-definite matrix of shape `n x n`,
            /// replicated `batch` times.
            fn spd<T: Scalar>(n: usize, batch: usize, order: Order, seed: u64) -> Vec<T> {
                let m = fill_seeded::<T>(n * n, seed);
                let mut mm = vec![T::zero(); n * n];
                mul_adj_b(n, n, n, order, &m, &m, &mut mm);
                for i in 0..n {
                    mm[idx(i, i, n, n, order)].add_assign(T::from_f64(n as f64 + 1.0));
                }
                replicate(&mm, batch)
            }

            // ------------------------------------------- in_place routes (one body, both faer rows)

            // `lu`, `cholesky` and `solve` are timed twice on each faer row. The op with an empty
            // note is the owning route (`PartialPivLu::new`, `Llt::new`, `partial_piv_lu().solve()`
            // upstream; the fork's composite rebuild of them). The op noted `in_place` is this text,
            // expanded identically for both crates: the matrix (and right-hand side), the
            // permutations and a `MemStack` sized by the library's own `*_scratch` are allocated
            // once, before the clock; `prepare` restores the input into them; the timed call is the
            // in-place factorization (and solve) alone, on the input's own layout (a row-major
            // input is factored where it is). The vendor row's `lu`, `cholesky` and `solve` are the
            // same kind of call. See docs/reliable-measurement.md, "Allocation and the clock".

            /// `(L, U)` of a packed LU, built element by element for the check (outside the clock).
            fn check_split_lu<T: ComplexField>(lu: MatRef<'_, T>) -> (fk::Mat<T>, fk::Mat<T>) {
                let (m, n) = lu.shape();
                let size = m.min(n);
                let mut l = fk::Mat::<T>::zeros(m, size);
                let mut u = fk::Mat::<T>::zeros(size, n);
                for j in 0..size {
                    for i in j + 1..m {
                        l[(i, j)] = lu[(i, j)].clone();
                    }
                    l[(j, j)] = T::one_impl();
                }
                for j in 0..n {
                    for i in 0..=j.min(size - 1) {
                        u[(i, j)] = lu[(i, j)].clone();
                    }
                }
                (l, u)
            }

            /// `max |(P A)_{ij} - (L U)_{ij}|` with `(P A)` row `i` = `A` row `fwd[i]`.
            fn lu_residual<T: Scalar + ComplexField>(
                a0: &[T], lu: MatRef<'_, T>, fwd: &[usize], m: usize, n: usize, order: Order,
            ) -> f64 {
                let (l, u) = check_split_lu(lu);
                let size = m.min(n);
                let l = to_layout(l.as_ref(), m, size, order);
                let u = to_layout(u.as_ref(), size, n, order);
                let mut prod = vec![T::zero(); m * n];
                mul(m, size, n, order, &l, &u, &mut prod);
                let mut pa = vec![T::zero(); m * n];
                for i in 0..m {
                    for j in 0..n {
                        pa[idx(i, j, m, n, order)] = a0[idx(fwd[i], j, m, n, order)];
                    }
                }
                max_diff(&pa, &prod)
            }

            fn lu_in_place_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                use fk::linalg::lu::partial_pivoting::factor as plu;
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 5));
                struct S<T: ComplexField> {
                    a0: Vec<T>,
                    a: Vec<T>,
                    perms: Vec<(Vec<usize>, Vec<usize>)>,
                    mem: MemBuffer,
                    m: usize,
                    n: usize,
                    batch: usize,
                    order: Order,
                    par: ParCfg,
                }
                let par = cfg_par();
                let req = plu::lu_in_place_scratch::<usize, T>(m, n, parmod::shape(par), Default::default());
                let state = S {
                    a: a0.clone(),
                    a0,
                    perms: (0..batch).map(|_| (vec![0usize; m], vec![0usize; m])).collect(),
                    mem: MemBuffer::new(req),
                    m,
                    n,
                    batch,
                    order,
                    par,
                };
                Ok(op_from(
                    state,
                    "in_place".to_owned(),
                    |s| {
                        s.a.copy_from_slice(&s.a0);
                        for (fwd, bwd) in &mut s.perms {
                            fwd.fill(0);
                            bwd.fill(0);
                        }
                        Ok(())
                    },
                    |s, mut tk| {
                        let len = s.m * s.n;
                        for bi in 0..s.batch {
                            let (fwd, bwd) = &mut s.perms[bi];
                            plu::lu_in_place(
                                mmut(&mut s.a[bi * len..(bi + 1) * len], s.m, s.n, s.order),
                                fwd,
                                bwd,
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem),
                                Default::default(),
                            );
                        }
                        Ok(())
                    },
                    |s| {
                        let len = s.m * s.n;
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * len..(bi + 1) * len];
                            let lu = mref(&s.a[bi * len..(bi + 1) * len], s.m, s.n, s.order);
                            let r = lu_residual(a0, lu, &s.perms[bi].0, s.m, s.n, s.order);
                            within::<T>(r, frob(a0), s.n.max(s.m), "lu in_place")?;
                        }
                        Ok(())
                    },
                ))
            }

            /// Only the lower triangle is read and written; the upper keeps the input, as `potrf`'s.
            fn cholesky_in_place_op<T: Scalar + ComplexField>(
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                use fk::linalg::cholesky::llt::factor as llt;
                let a0 = spd::<T>(n, batch, order, seed(n, n, 11));
                struct S<T: ComplexField> {
                    a0: Vec<T>,
                    a: Vec<T>,
                    mem: MemBuffer,
                    n: usize,
                    batch: usize,
                    order: Order,
                    par: ParCfg,
                }
                let par = cfg_par();
                let req = llt::cholesky_in_place_scratch::<T>(n, parmod::shape(par), Default::default());
                let state = S { a: a0.clone(), a0, mem: MemBuffer::new(req), n, batch, order, par };
                Ok(op_from(
                    state,
                    "in_place".to_owned(),
                    |s| {
                        s.a.copy_from_slice(&s.a0);
                        Ok(())
                    },
                    |s, mut tk| {
                        let len = s.n * s.n;
                        for bi in 0..s.batch {
                            llt::cholesky_in_place(
                                mmut(&mut s.a[bi * len..(bi + 1) * len], s.n, s.n, s.order),
                                Default::default(),
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem),
                                Default::default(),
                            )
                            .map_err(|e| format!("llt in_place failed: {e:?}"))?;
                        }
                        Ok(())
                    },
                    |s| {
                        let len = s.n * s.n;
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * len..(bi + 1) * len];
                            let a = &s.a[bi * len..(bi + 1) * len];
                            let mut l = vec![T::zero(); len];
                            for j in 0..s.n {
                                for i in j..s.n {
                                    l[idx(i, j, s.n, s.n, s.order)] = a[idx(i, j, s.n, s.n, s.order)];
                                }
                            }
                            let mut recon = vec![T::zero(); len];
                            mul_adj_b(s.n, s.n, s.n, s.order, &l, &l, &mut recon);
                            within::<T>(max_diff(a0, &recon), frob(a0), s.n, "cholesky in_place")?;
                        }
                        Ok(())
                    },
                ))
            }

            /// `lu_in_place`, then `solve_in_place_with_conj` with the packed factor passed as both
            /// `L` (its unit-lower part is read) and `U` (its upper part).
            fn solve_in_place_op<T: Scalar + ComplexField>(
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                use fk::linalg::lu::partial_pivoting::{factor as plu, solve as plu_solve};
                let a0 = diag_dominant::<T>(n, batch, order, seed(n, n, 13));
                let b0 = fill_seeded::<T>(n * batch, seed(n, 1, 17));
                struct S<T: ComplexField> {
                    a0: Vec<T>,
                    a: Vec<T>,
                    b0: Vec<T>,
                    b: Vec<T>,
                    perms: Vec<(Vec<usize>, Vec<usize>)>,
                    mem_lu: MemBuffer,
                    mem_solve: MemBuffer,
                    sink: T,
                    n: usize,
                    batch: usize,
                    order: Order,
                    par: ParCfg,
                }
                let par = cfg_par();
                let state = S {
                    a: a0.clone(),
                    a0,
                    b: b0.clone(),
                    b0,
                    perms: (0..batch).map(|_| (vec![0usize; n], vec![0usize; n])).collect(),
                    mem_lu: MemBuffer::new(plu::lu_in_place_scratch::<usize, T>(n, n, parmod::shape(par), Default::default())),
                    mem_solve: MemBuffer::new(plu_solve::solve_in_place_scratch::<usize, T>(n, 1, parmod::shape(par))),
                    sink: T::zero(),
                    n,
                    batch,
                    order,
                    par,
                };
                Ok(op_from(
                    state,
                    "in_place".to_owned(),
                    |s| {
                        s.a.copy_from_slice(&s.a0);
                        s.b.copy_from_slice(&s.b0);
                        Ok(())
                    },
                    |s, mut tk| {
                        let len = s.n * s.n;
                        for bi in 0..s.batch {
                            let (fwd, bwd) = &mut s.perms[bi];
                            let a = &mut s.a[bi * len..(bi + 1) * len];
                            let (_, perm) = plu::lu_in_place(
                                mmut(a, s.n, s.n, s.order),
                                fwd,
                                bwd,
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem_lu),
                                Default::default(),
                            );
                            let lu = mref(a, s.n, s.n, s.order);
                            plu_solve::solve_in_place_with_conj(
                                lu,
                                lu,
                                perm,
                                Conj::No,
                                mmut(&mut s.b[bi * s.n..(bi + 1) * s.n], s.n, 1, s.order),
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem_solve),
                            );
                            acc(&mut s.sink, s.b[bi * s.n].clone());
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
                            within::<T>(max_diff(&ax, b0), frob(b0), s.n, "solve in_place")?;
                        }
                        Ok(())
                    },
                ))
            }

            // ------------------------------------------ full_piv_lu, svd, eigh: the same rule

            // `full_piv_lu`, `svd_values`, `svd_thin` and `eigh` are timed twice on each faer row, as
            // `lu` is: the empty-note op is the owning route (the library's allocation of its outputs
            // and scratch is its cost), the `in_place` op below is this text expanded for both crates.
            // Everything the harness needs is allocated once before the clock: the factor matrix and
            // the four permutations (restored in `prepare`) for `full_piv_lu`; the outputs `S`, `U`,
            // `V` and the `MemStack` sized by the library's own `*_scratch` for the decompositions,
            // which read `A` without modifying it, so they run straight on the input and need no
            // restore. `prepare` zeroes the reused outputs, so a skipped execution cannot pass its check.

            /// `max |(P A Q)_{ij} - (L U)_{ij}|` with `(P A Q)` entry `(i, j)` = `A` entry `(rf[i], cf[j])`.
            fn full_piv_residual<T: Scalar + ComplexField>(
                a0: &[T], lu: MatRef<'_, T>, rf: &[usize], cf: &[usize], m: usize, n: usize, order: Order,
            ) -> f64 {
                let (l, u) = check_split_lu(lu);
                let size = m.min(n);
                let l = to_layout(l.as_ref(), m, size, order);
                let u = to_layout(u.as_ref(), size, n, order);
                let mut prod = vec![T::zero(); m * n];
                mul(m, size, n, order, &l, &u, &mut prod);
                let mut paq = vec![T::zero(); m * n];
                for i in 0..m {
                    for j in 0..n {
                        paq[idx(i, j, m, n, order)] = a0[idx(rf[i], cf[j], m, n, order)];
                    }
                }
                max_diff(&paq, &prod)
            }

            fn full_piv_lu_in_place_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                use fk::linalg::lu::full_pivoting::factor as flu;
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 7));
                struct S<T: ComplexField> {
                    a0: Vec<T>,
                    a: Vec<T>,
                    perms: Vec<[Vec<usize>; 4]>,
                    mem: MemBuffer,
                    m: usize,
                    n: usize,
                    batch: usize,
                    order: Order,
                    par: ParCfg,
                }
                let par = cfg_par();
                let req = flu::lu_in_place_scratch::<usize, T>(m, n, parmod::shape(par), Default::default());
                let state = S {
                    a: a0.clone(),
                    a0,
                    perms: (0..batch).map(|_| [vec![0usize; m], vec![0usize; m], vec![0usize; n], vec![0usize; n]]).collect(),
                    mem: MemBuffer::new(req),
                    m,
                    n,
                    batch,
                    order,
                    par,
                };
                Ok(op_from(
                    state,
                    "in_place".to_owned(),
                    |s| {
                        s.a.copy_from_slice(&s.a0);
                        for p in &mut s.perms {
                            for v in p.iter_mut() {
                                v.fill(0);
                            }
                        }
                        Ok(())
                    },
                    |s, mut tk| {
                        let len = s.m * s.n;
                        for bi in 0..s.batch {
                            let [rf, rb, cf, cb] = &mut s.perms[bi];
                            flu::lu_in_place(
                                mmut(&mut s.a[bi * len..(bi + 1) * len], s.m, s.n, s.order),
                                rf,
                                rb,
                                cf,
                                cb,
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem),
                                Default::default(),
                            );
                        }
                        Ok(())
                    },
                    |s| {
                        let len = s.m * s.n;
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * len..(bi + 1) * len];
                            let lu = mref(&s.a[bi * len..(bi + 1) * len], s.m, s.n, s.order);
                            let [rf, _, cf, _] = &s.perms[bi];
                            let r = full_piv_residual(a0, lu, rf, cf, s.m, s.n, s.order);
                            within::<T>(r, frob(a0), s.n.max(s.m), "full_piv_lu in_place")?;
                        }
                        Ok(())
                    },
                ))
            }

            /// `svd` with prepared `S` (and `U`, `V` when `thin`) and a prepared `MemStack`.
            fn svd_in_place_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
                thin: bool,
            ) -> Result<Op, String>
            where
                T::Real: Scalar,
            {
                use fk::linalg::svd::ComputeSvdVectors as Cv;
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, if thin { 31 } else { 29 }));
                let r = m.min(n);
                struct S<T: ComplexField> {
                    a0: Vec<T>,
                    sv: Vec<fk::diag::Diag<T>>,
                    u: Vec<fk::Mat<T>>,
                    v: Vec<fk::Mat<T>>,
                    mem: MemBuffer,
                    m: usize,
                    n: usize,
                    batch: usize,
                    order: Order,
                    thin: bool,
                    par: ParCfg,
                }
                let par = cfg_par();
                let mode = if thin { Cv::Thin } else { Cv::No };
                let req = fk::linalg::svd::svd_scratch::<T>(m, n, mode, mode, parmod::shape(par), Default::default());
                let state = S {
                    a0,
                    sv: (0..batch).map(|_| fk::diag::Diag::<T>::zeros(r)).collect(),
                    u: (0..if thin { batch } else { 0 }).map(|_| fk::Mat::<T>::zeros(m, r)).collect(),
                    v: (0..if thin { batch } else { 0 }).map(|_| fk::Mat::<T>::zeros(n, r)).collect(),
                    mem: MemBuffer::new(req),
                    m,
                    n,
                    batch,
                    order,
                    thin,
                    par,
                };
                Ok(op_from(
                    state,
                    "in_place".to_owned(),
                    |s| {
                        for d in &mut s.sv {
                            d.as_mut().column_vector_mut().fill(T::zero_impl());
                        }
                        for x in s.u.iter_mut().chain(s.v.iter_mut()) {
                            x.as_mut().fill(T::zero_impl());
                        }
                        Ok(())
                    },
                    |s, mut tk| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let (u, v) = if s.thin {
                                (Some(s.u[bi].as_mut()), Some(s.v[bi].as_mut()))
                            } else {
                                (None, None)
                            };
                            fk::linalg::svd::svd(
                                mref(a0, s.m, s.n, s.order),
                                s.sv[bi].as_mut(),
                                u,
                                v,
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem),
                                Default::default(),
                            )
                            .map_err(|e| format!("svd in_place failed: {e:?}"))?;
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            if !s.thin {
                                let sv: Vec<T::Real> =
                                    s.sv[bi].column_vector().iter().map(|x| T::real_part_impl(x)).collect();
                                check_singular_values::<T>(a0, &sv, s.m, s.n)?;
                                continue;
                            }
                            let r = s.sv[bi].dim();
                            let u = to_layout(s.u[bi].as_ref(), s.m, r, s.order);
                            let v = to_layout(s.v[bi].as_ref(), s.n, r, s.order);
                            let scale = frob(a0);
                            within::<T>(gram_residual(&u, s.m, r, s.order), 1.0, s.m, "svd_thin U")?;
                            within::<T>(gram_residual(&v, s.n, r, s.order), 1.0, s.n, "svd_thin V")?;
                            let s_diag = diag_vec(s.sv[bi].as_ref());
                            let scaled_v = scale_cols(&v, s.n, r, &s_diag, s.order);
                            let mut recon = vec![T::zero(); s.m * s.n];
                            mul_adj_b(s.m, r, s.n, s.order, &u, &scaled_v, &mut recon);
                            within::<T>(max_diff(a0, &recon), scale, s.n.max(s.m), "svd_thin")?;
                        }
                        Ok(())
                    },
                ))
            }

            /// `self_adjoint_evd` (lower triangle) with prepared eigenvalues, `U` and `MemStack`.
            fn eigh_in_place_op<T: Scalar + ComplexField>(
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let a0 = hermitian::<T>(n, order, seed(n, n, 37));
                let a0 = replicate(&a0, batch);
                struct S<T: ComplexField> {
                    a0: Vec<T>,
                    sv: Vec<fk::diag::Diag<T>>,
                    u: Vec<fk::Mat<T>>,
                    mem: MemBuffer,
                    n: usize,
                    batch: usize,
                    order: Order,
                    par: ParCfg,
                }
                let par = cfg_par();
                let req = fk::linalg::evd::self_adjoint_evd_scratch::<T>(
                    n,
                    fk::linalg::evd::ComputeEigenvectors::Yes,
                    parmod::shape(par),
                    Default::default(),
                );
                let state = S {
                    a0,
                    sv: (0..batch).map(|_| fk::diag::Diag::<T>::zeros(n)).collect(),
                    u: (0..batch).map(|_| fk::Mat::<T>::zeros(n, n)).collect(),
                    mem: MemBuffer::new(req),
                    n,
                    batch,
                    order,
                    par,
                };
                Ok(op_from(
                    state,
                    "in_place".to_owned(),
                    |s| {
                        for d in &mut s.sv {
                            d.as_mut().column_vector_mut().fill(T::zero_impl());
                        }
                        for x in &mut s.u {
                            x.as_mut().fill(T::zero_impl());
                        }
                        Ok(())
                    },
                    |s, mut tk| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                            fk::linalg::evd::self_adjoint_evd(
                                mref(a0, s.n, s.n, s.order),
                                s.sv[bi].as_mut(),
                                Some(s.u[bi].as_mut()),
                                parmod::mk_from(&mut tk, s.par),
                                MemStack::new(&mut s.mem),
                                Default::default(),
                            )
                            .map_err(|e| format!("self_adjoint_evd in_place failed: {e:?}"))?;
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                            let u = to_layout(s.u[bi].as_ref(), s.n, s.n, s.order);
                            let scale = frob(a0);
                            within::<T>(gram_residual(&u, s.n, s.n, s.order), 1.0, s.n, "eigh U")?;
                            let s_diag = diag_vec(s.sv[bi].as_ref());
                            let mut au = vec![T::zero(); s.n * s.n];
                            mul(s.n, s.n, s.n, s.order, a0, &u, &mut au);
                            let us = scale_cols(&u, s.n, s.n, &s_diag, s.order);
                            within::<T>(max_diff(&au, &us), scale, s.n, "eigh in_place")?;
                        }
                        Ok(())
                    },
                ))
            }
        }
    };
}

/// The upstream row's families: the high level methods, as the campaign has always measured them,
/// reading upstream's own global parallelism inside the held spindle group. Written once against
/// `$krate`'s solver types, so it is only expanded for upstream (the fork's 0.25 methods take a
/// `par` and are replaced by `explicit_ops!`).
macro_rules! highlevel_ops {
    () => {
        use fk::linalg::solvers::{
            DenseSolveCore, FullPivLu, Llt, PartialPivLu, Qr, SelfAdjointEigen, Solve, Svd,
        };
        use fk::Side;
            // -------------------------------------------------------- qr (composite)

            fn qr_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 3));
                let size = m.min(n);
                struct S<T> {
                    a0: Vec<T>,
                    outputs: Vec<(Qr<T>, fk::Mat<T>)>,
                    stash: Vec<(Qr<T>, fk::Mat<T>)>,
                    m: usize,
                    n: usize,
                    size: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    a0,
                    outputs: Vec::with_capacity(batch),
                    stash: Vec::new(),
                    m,
                    n,
                    size,
                    batch,
                    order,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| {
                        s.stash.append(&mut s.outputs);
                        Ok(())
                    },
                    |s, _tk| {
                        s.stash.clear(); // the previous call's outputs are freed inside the clock
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            // The composite the vendor route's `orgqr` arm matches:
                            // factor, then form the compact `m x min(m, n)` Q. Both the
                            // actual Q and R this call returns are retained, so the
                            // check below sees the run's own output.
                            let qr = mref(a0, s.m, s.n, s.order).qr();
                            let q = qr.compute_thin_Q();
                            s.outputs.push((qr, q));
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let (qr, q) = &s.outputs[bi];
                            let q = to_layout(q.as_ref(), s.m, s.size, s.order);
                            let r = to_layout(qr.thin_R(), s.size, s.n, s.order);
                            let mut recon = vec![T::zero(); s.m * s.n];
                            mul(s.m, s.size, s.n, s.order, &q, &r, &mut recon);
                            within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "qr")?;
                            // Same acceptance criterion as `qr_factor`: the compact Q is
                            // the factor that multiplies R, and the full Q's remaining
                            // columns are an arbitrary completion.
                            within::<T>(
                                gram_residual(&q, s.m, s.size, s.order),
                                1.0,
                                s.m,
                                "qr Q",
                            )?;
                        }
                        Ok(())
                    },
                ))
            }

            // ------------------------------------------------------------ lu

            fn lu_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 5));
                struct S<T> {
                    a0: Vec<T>,
                    lus: Vec<PartialPivLu<T>>,
                    stash: Vec<PartialPivLu<T>>,
                    m: usize,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    a0,
                    lus: Vec::with_capacity(batch),
                    stash: Vec::new(),
                    m,
                    n,
                    batch,
                    order,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| {
                        s.stash.append(&mut s.lus);
                        Ok(())
                    },
                    |s, _tk| {
                        s.stash.clear(); // the previous call's outputs are freed inside the clock
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let lu = mref(a0, s.m, s.n, s.order).partial_piv_lu();
                            s.lus.push(lu);
                        }
                        Ok(())
                    },
                    |s| {
                        // The reconstruct is computed from the actual factorization
                        // object the timed call returned, not from a fresh one.
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let recon = to_layout(s.lus[bi].reconstruct().as_ref(), s.m, s.n, s.order);
                            within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "lu")?;
                        }
                        Ok(())
                    },
                ))
            }

            fn full_piv_lu_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 7));
                struct S<T> {
                    a0: Vec<T>,
                    lus: Vec<FullPivLu<T>>,
                    stash: Vec<FullPivLu<T>>,
                    m: usize,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    a0,
                    lus: Vec::with_capacity(batch),
                    stash: Vec::new(),
                    m,
                    n,
                    batch,
                    order,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| {
                        s.stash.append(&mut s.lus);
                        Ok(())
                    },
                    |s, _tk| {
                        s.stash.clear(); // the previous call's outputs are freed inside the clock
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let lu = mref(a0, s.m, s.n, s.order).full_piv_lu();
                            s.lus.push(lu);
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let recon = to_layout(s.lus[bi].reconstruct().as_ref(), s.m, s.n, s.order);
                            within::<T>(
                                max_diff(a0, &recon),
                                frob(a0),
                                s.n.max(s.m),
                                "full_piv_lu",
                            )?;
                        }
                        Ok(())
                    },
                ))
            }

            // ------------------------------------------------------ cholesky

            fn cholesky_op<T: Scalar + ComplexField>(
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let a0 = spd::<T>(n, batch, order, seed(n, n, 11));
                struct S<T> {
                    a0: Vec<T>,
                    outputs: Vec<Llt<T>>,
                    stash: Vec<Llt<T>>,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    a0,
                    outputs: Vec::with_capacity(batch),
                    stash: Vec::new(),
                    n,
                    batch,
                    order,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| { s.stash.append(&mut s.outputs); Ok(()) },
                    |s, _tk| {
                        s.stash.clear(); // the previous call's outputs are freed inside the clock
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                            let llt = mref(a0, s.n, s.n, s.order)
                                .llt(Side::Lower)
                                .map_err(|e| format!("llt failed: {e:?}"))?;
                            s.outputs.push(llt);
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                            let recon = to_layout(s.outputs[bi].reconstruct().as_ref(), s.n, s.n, s.order);
                            within::<T>(max_diff(a0, &recon), frob(a0), s.n, "cholesky")?;
                        }
                        Ok(())
                    },
                ))
            }

            // ----------------------------------------------------------- solve

            fn solve_op<T: Scalar + ComplexField>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
                let a0 = diag_dominant::<T>(n, batch, order, seed(n, n, 13));
                let b0 = fill_seeded::<T>(n * batch, seed(n, 1, 17));
                struct S<T> {
                    a0: Vec<T>,
                    b0: Vec<T>,
                    b: Vec<T>,
                    sink: T,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
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
                        Ok(())
                    },
                    |s, _tk| {
                        for bi in 0..s.batch {
                            let a = mref(&s.a0[bi * s.n * s.n..], s.n, s.n, s.order);
                            let lu = a.partial_piv_lu();
                            let rhs = mmut(&mut s.b[bi * s.n..], s.n, 1, s.order);
                            lu.solve_in_place(rhs);
                            acc(&mut s.sink, s.b[bi * s.n].clone());
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

            // -------------------------------------------------- triangular_solve

            fn triangular_solve_op<T: Scalar + ComplexField>(
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
                    sink: T,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    u0,
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
                        Ok(())
                    },
                    |s, _tk| {
                        for bi in 0..s.batch {
                            let u = mref(&s.u0[bi * s.n * s.n..], s.n, s.n, s.order);
                            let rhs = mmut(&mut s.b[bi * s.n..], s.n, 1, s.order);
                            u.solve_upper_triangular_in_place(rhs);
                            acc(&mut s.sink, s.b[bi * s.n].clone());
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

            // ----------------------------------------------------- svd_values

            fn svd_values_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String>
            where
                T::Real: Scalar,
            {
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 29));
                struct S<T: ComplexField> {
                    a0: Vec<T>,
                    outputs: Vec<Vec<T::Real>>,
                    stash: Vec<Vec<T::Real>>,
                    m: usize,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    a0,
                    outputs: Vec::with_capacity(batch),
                    stash: Vec::new(),
                    m,
                    n,
                    batch,
                    order,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| { s.stash.append(&mut s.outputs); Ok(()) },
                    |s, _tk| {
                        s.stash.clear(); // the previous call's outputs are freed inside the clock
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let sv = mref(a0, s.m, s.n, s.order)
                                .singular_values()
                                .map_err(|e| format!("singular_values failed: {e:?}"))?;
                            s.outputs.push(sv);
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let sv = &s.outputs[bi];
                            let r = sv.len();
                            let eps = tol_of::<T>(s.n.max(s.m));
                            let neg_eps = from_f64::<T::Real>(-eps);
                            let eps_r = from_f64::<T::Real>(eps);
                            for i in 0..r {
                                // Signed, so a sign-flipped spectrum is caught here rather than
                                // passing the ordering and norm checks below.
                                let value = <T::Real as Scalar>::abs(sv[i]);
                                if !value.is_finite() || sv[i] < neg_eps {
                                    return Err(format!("singular value {i} is negative: {value}"));
                                }
                                if i + 1 < r && sv[i] < sv[i + 1] - eps_r {
                                    return Err("singular values are not sorted".to_owned());
                                }
                            }
                            let sum: f64 = sv
                                .iter()
                                .map(|x| {
                                    let v = <T::Real as Scalar>::abs(*x);
                                    v * v
                                })
                                .sum();
                            let scale = frob(a0);
                            within::<T>(
                                (sum.sqrt() - scale).abs(),
                                scale,
                                s.n.max(s.m),
                                "svd_values norm identity",
                            )?;
                        }
                        Ok(())
                    },
                ))
            }

            // ------------------------------------------------------- svd_thin

            fn svd_thin_op<T: Scalar + ComplexField>(
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
            ) -> Result<Op, String> {
                let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 31));
                struct S<T> {
                    a0: Vec<T>,
                    outputs: Vec<Svd<T>>,
                    stash: Vec<Svd<T>>,
                    m: usize,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    a0,
                    outputs: Vec::with_capacity(batch),
                    stash: Vec::new(),
                    m,
                    n,
                    batch,
                    order,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| { s.stash.append(&mut s.outputs); Ok(()) },
                    |s, _tk| {
                        s.stash.clear(); // the previous call's outputs are freed inside the clock
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let svd = mref(a0, s.m, s.n, s.order)
                                .thin_svd()
                                .map_err(|e| format!("thin_svd failed: {e:?}"))?;
                            s.outputs.push(svd);
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                            let svd = &s.outputs[bi];
                            let r = svd.S().dim();
                            let u = to_layout(svd.U(), s.m, r, s.order);
                            let v = to_layout(svd.V(), s.n, r, s.order);
                            let scale = frob(a0);
                            within::<T>(gram_residual(&u, s.m, r, s.order), 1.0, s.m, "svd_thin U")?;
                            within::<T>(gram_residual(&v, s.n, r, s.order), 1.0, s.n, "svd_thin V")?;
                            let s_diag = diag_vec(svd.S());
                            let scaled_v = scale_cols(&v, s.n, r, &s_diag, s.order);
                            let mut recon = vec![T::zero(); s.m * s.n];
                            mul_adj_b(s.m, r, s.n, s.order, &u, &scaled_v, &mut recon);
                            within::<T>(max_diff(a0, &recon), scale, s.n.max(s.m), "svd_thin")?;
                        }
                        Ok(())
                    },
                ))
            }

            // ---------------------------------------------------------- eigh

            fn eigh_op<T: Scalar + ComplexField>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
                let a0 = hermitian::<T>(n, order, seed(n, n, 37));
                let a0 = replicate(&a0, batch);
                struct S<T> {
                    a0: Vec<T>,
                    outputs: Vec<SelfAdjointEigen<T>>,
                    stash: Vec<SelfAdjointEigen<T>>,
                    n: usize,
                    batch: usize,
                    order: Order,
                }
                let state = S {
                    a0,
                    outputs: Vec::with_capacity(batch),
                    stash: Vec::new(),
                    n,
                    batch,
                    order,
                };
                Ok(op_from(
                    state,
                    String::new(),
                    |s| { s.stash.append(&mut s.outputs); Ok(()) },
                    |s, _tk| {
                        s.stash.clear(); // the previous call's outputs are freed inside the clock
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                            let evd = mref(a0, s.n, s.n, s.order)
                                .self_adjoint_eigen(Side::Lower)
                                .map_err(|e| format!("self_adjoint_eigen failed: {e:?}"))?;
                            s.outputs.push(evd);
                        }
                        Ok(())
                    },
                    |s| {
                        for bi in 0..s.batch {
                            let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                            let evd = &s.outputs[bi];
                            let u = to_layout(evd.U(), s.n, s.n, s.order);
                            let scale = frob(a0);
                            within::<T>(gram_residual(&u, s.n, s.n, s.order), 1.0, s.n, "eigh U")?;
                            let s_diag = diag_vec(evd.S());
                            let mut au = vec![T::zero(); s.n * s.n];
                            mul(s.n, s.n, s.n, s.order, a0, &u, &mut au);
                            let us = scale_cols(&u, s.n, s.n, &s_diag, s.order);
                            within::<T>(max_diff(&au, &us), scale, s.n, "eigh")?;
                        }
                        Ok(())
                    },
                ))
            }

    };
}


/// The families that the fork rows run through explicit-parallelism entry points only.
///
/// `faer-fork` is measured at two API generations: the baseline, where the high level methods
/// (`partial_piv_lu()`, `thin_svd()`, ...) read a process-wide parallelism setting, and the
/// 0.25 candidate, where that setting is gone and the same methods take a `Par<'team>`. A call
/// through a high level method therefore cannot be written once for both. Every family below is
/// the composite the high level method performs, rebuilt from the low-level entry points that
/// take the row's parallelism as an argument, so the same text builds and measures the same work
/// against both revisions: the same copy of the input into an owned column-major matrix, the same
/// factorisation call, the same split of the packed factors, the same allocation of outputs.
/// The result of the timed call is kept, and the check inspects it (reconstructing with the
/// library's own low-level `reconstruct`, run sequentially, outside the clock).
macro_rules! explicit_ops {
    () => {
        use fk::linalg::cholesky::llt::{factor as llt_factor, reconstruct as llt_rec};
        use fk::linalg::lu::full_pivoting::{factor as flu, reconstruct as flu_rec};
        use fk::linalg::lu::partial_pivoting::{
            factor as plu, reconstruct as plu_rec, solve as plu_solve,
        };
        use fk::perm::PermRef;

        /// `(L, U)` of a packed LU, as `PartialPivLu::new` / `FullPivLu::new` split it: `L` is unit
        /// lower trapezoidal, `U` upper trapezoidal.
        fn split_lu<T: ComplexField>(lu: fk::Mat<T>) -> (fk::Mat<T>, fk::Mat<T>) {
            let (m, n) = lu.shape();
            let size = m.min(n);
            if m >= n {
                let mut u = fk::Mat::<T>::zeros(size, size);
                for j in 0..size {
                    for i in 0..=j {
                        u[(i, j)] = lu[(i, j)].clone();
                    }
                }
                let mut l = lu;
                for j in 0..n {
                    for i in 0..j {
                        l[(i, j)] = T::zero_impl();
                    }
                    l[(j, j)] = T::one_impl();
                }
                (l, u)
            } else {
                let mut l = fk::Mat::<T>::zeros(size, size);
                for j in 0..size {
                    for i in j + 1..size {
                        l[(i, j)] = lu[(i, j)].clone();
                    }
                    l[(j, j)] = T::one_impl();
                }
                let mut u = lu;
                for j in 0..size {
                    for i in j + 1..m {
                        u[(i, j)] = T::zero_impl();
                    }
                }
                (l, u)
            }
        }

        /// The factorization a timed call measures, so it runs on the row's token: `verify_par`
        /// is `Par::Seq` under `fork-team`, and handing it here made the fork `lu` and `solve`
        /// rows sequential at every thread count.
        fn partial_lu_of<T: ComplexField>(
            a: MatRef<'_, T>,
            tk: &mut crate::op::Token<'_>,
            cfg: ParCfg,
        ) -> (fk::Mat<T>, fk::Mat<T>, Vec<usize>, Vec<usize>) {
            let mut lu = a.to_owned();
            let (m, n) = lu.shape();
            let mut fwd = vec![0usize; m];
            let mut bwd = vec![0usize; m];
            let req = plu::lu_in_place_scratch::<usize, T>(m, n, parmod::shape(cfg), Default::default());
            plu::lu_in_place(
                lu.as_mut(),
                &mut fwd,
                &mut bwd,
                parmod::mk_from(tk, cfg),
                MemStack::new(&mut MemBuffer::new(req)),
                Default::default(),
            );
            let (l, u) = split_lu(lu);
            (l, u, fwd, bwd)
        }

        // ------------------------------------------------------- qr (composite)

        fn qr_op<T: Scalar + ComplexField>(
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
        ) -> Result<Op, String> {
            let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 3));
            let size = m.min(n);
            struct S<T: ComplexField> {
                a0: Vec<T>,
                outputs: Vec<(fk::Mat<T>, fk::Mat<T>)>,
                stash: Vec<(fk::Mat<T>, fk::Mat<T>)>,
                m: usize,
                n: usize,
                size: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                outputs: Vec::with_capacity(batch),
                stash: Vec::new(),
                m,
                n,
                size,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.stash.append(&mut s.outputs);
                    Ok(())
                },
                |s, mut tk| {
                    s.stash.clear(); // the previous call's outputs are freed inside the clock
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        // `Qr::new` then `compute_thin_Q`, with the row's parallelism.
                        let mut qr = mref(a0, s.m, s.n, s.order).to_owned();
                        let block_size = qrf::recommended_block_size::<T>(s.m, s.n);
                        let mut coeff = fk::Mat::<T>::zeros(block_size, s.size);
                        let req = qrf::qr_in_place_scratch::<T>(
                            s.m,
                            s.n,
                            block_size,
                            parmod::shape(s.par),
                            Default::default(),
                        );
                        qrf::qr_in_place(
                            qr.as_mut(),
                            coeff.as_mut(),
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(req)),
                            Default::default(),
                        );
                        let (basis, r) = split_lu(qr);
                        let mut q = fk::Mat::<T>::identity(s.m, s.size);
                        householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
                            basis.as_ref(),
                            coeff.as_ref(),
                            Conj::No,
                            q.as_mut(),
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(
                                householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<T>(
                                    s.m, block_size, s.size,
                                ),
                            )),
                        );
                        s.outputs.push((r, q));
                    }
                    Ok(())
                },
                |s| {
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        let (r, q) = &s.outputs[bi];
                        let q = to_layout(q.as_ref(), s.m, s.size, s.order);
                        let r = to_layout(r.as_ref(), s.size, s.n, s.order);
                        let mut recon = vec![T::zero(); s.m * s.n];
                        mul(s.m, s.size, s.n, s.order, &q, &r, &mut recon);
                        within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "qr")?;
                        within::<T>(gram_residual(&q, s.m, s.size, s.order), 1.0, s.m, "qr Q")?;
                    }
                    Ok(())
                },
            ))
        }

        // ------------------------------------------------------------ lu

        fn lu_op<T: Scalar + ComplexField>(
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
        ) -> Result<Op, String> {
            let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 5));
            struct S<T: ComplexField> {
                a0: Vec<T>,
                lus: Vec<(fk::Mat<T>, fk::Mat<T>, Vec<usize>, Vec<usize>)>,
                stash: Vec<(fk::Mat<T>, fk::Mat<T>, Vec<usize>, Vec<usize>)>,
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                lus: Vec::with_capacity(batch),
                stash: Vec::new(),
                m,
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.stash.append(&mut s.lus);
                    Ok(())
                },
                |s, mut tk| {
                    s.stash.clear(); // the previous call's outputs are freed inside the clock
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        s.lus.push(partial_lu_of(mref(a0, s.m, s.n, s.order), &mut tk, s.par));
                    }
                    Ok(())
                },
                |s| {
                    // The reconstruct is computed from the actual factorization the timed call
                    // returned, not from a fresh one.
                    let seq = ParCfg::Seq;
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        let (l, u, fwd, bwd) = &s.lus[bi];
                        let mut out = fk::Mat::<T>::zeros(s.m, s.n);
                        plu_rec::reconstruct(
                            out.as_mut(),
                            l.as_ref(),
                            u.as_ref(),
                            PermRef::new_checked(fwd, bwd, s.m),
                            parmod::verify_par(seq),
                            MemStack::new(&mut MemBuffer::new(
                                plu_rec::reconstruct_scratch::<usize, T>(s.m, s.n, parmod::shape(seq)),
                            )),
                        );
                        let recon = to_layout(out.as_ref(), s.m, s.n, s.order);
                        within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "lu")?;
                    }
                    Ok(())
                },
            ))
        }

        fn full_piv_lu_op<T: Scalar + ComplexField>(
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
        ) -> Result<Op, String> {
            let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 7));
            struct S<T: ComplexField> {
                a0: Vec<T>,
                lus: Vec<(fk::Mat<T>, fk::Mat<T>, [Vec<usize>; 4])>,
                stash: Vec<(fk::Mat<T>, fk::Mat<T>, [Vec<usize>; 4])>,
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                lus: Vec::with_capacity(batch),
                stash: Vec::new(),
                m,
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.stash.append(&mut s.lus);
                    Ok(())
                },
                |s, mut tk| {
                    s.stash.clear(); // the previous call's outputs are freed inside the clock
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        let mut lu = mref(a0, s.m, s.n, s.order).to_owned();
                        let (m, n) = lu.shape();
                        let mut rf = vec![0usize; m];
                        let mut rb = vec![0usize; m];
                        let mut cf = vec![0usize; n];
                        let mut cb = vec![0usize; n];
                        let req = flu::lu_in_place_scratch::<usize, T>(
                            m,
                            n,
                            parmod::shape(s.par),
                            Default::default(),
                        );
                        flu::lu_in_place(
                            lu.as_mut(),
                            &mut rf,
                            &mut rb,
                            &mut cf,
                            &mut cb,
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(req)),
                            Default::default(),
                        );
                        let (l, u) = split_lu(lu);
                        s.lus.push((l, u, [rf, rb, cf, cb]));
                    }
                    Ok(())
                },
                |s| {
                    let seq = ParCfg::Seq;
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        let (l, u, [rf, rb, cf, cb]) = &s.lus[bi];
                        let mut out = fk::Mat::<T>::zeros(s.m, s.n);
                        flu_rec::reconstruct(
                            out.as_mut(),
                            l.as_ref(),
                            u.as_ref(),
                            PermRef::new_checked(rf, rb, s.m),
                            PermRef::new_checked(cf, cb, s.n),
                            parmod::verify_par(seq),
                            MemStack::new(&mut MemBuffer::new(
                                flu_rec::reconstruct_scratch::<usize, T>(s.m, s.n, parmod::shape(seq)),
                            )),
                        );
                        let recon = to_layout(out.as_ref(), s.m, s.n, s.order);
                        within::<T>(max_diff(a0, &recon), frob(a0), s.n.max(s.m), "full_piv_lu")?;
                    }
                    Ok(())
                },
            ))
        }

        // ------------------------------------------------------ cholesky

        fn cholesky_op<T: Scalar + ComplexField>(
            n: usize,
            batch: usize,
            order: Order,
        ) -> Result<Op, String> {
            let a0 = spd::<T>(n, batch, order, seed(n, n, 11));
            struct S<T: ComplexField> {
                a0: Vec<T>,
                outputs: Vec<fk::Mat<T>>,
                stash: Vec<fk::Mat<T>>,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                outputs: Vec::with_capacity(batch),
                stash: Vec::new(),
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.stash.append(&mut s.outputs);
                    Ok(())
                },
                |s, mut tk| {
                    s.stash.clear(); // the previous call's outputs are freed inside the clock
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                        // `Llt::new(A, Side::Lower)`: copy the lower triangle, factor, clear the upper.
                        let mut l = fk::Mat::<T>::zeros(s.n, s.n);
                        l.copy_from_triangular_lower(mref(a0, s.n, s.n, s.order));
                        let req = llt_factor::cholesky_in_place_scratch::<T>(
                            s.n,
                            parmod::shape(s.par),
                            Default::default(),
                        );
                        llt_factor::cholesky_in_place(
                            l.as_mut(),
                            Default::default(),
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(req)),
                            Default::default(),
                        )
                        .map_err(|e| format!("llt failed: {e:?}"))?;
                        for j in 1..s.n {
                            for i in 0..j {
                                l[(i, j)] = T::zero_impl();
                            }
                        }
                        s.outputs.push(l);
                    }
                    Ok(())
                },
                |s| {
                    let seq = ParCfg::Seq;
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                        let mut out = fk::Mat::<T>::zeros(s.n, s.n);
                        llt_rec::reconstruct(
                            out.as_mut(),
                            s.outputs[bi].as_ref(),
                            parmod::verify_par(seq),
                            MemStack::new(&mut MemBuffer::new(
                                llt_rec::reconstruct_scratch::<T>(s.n, parmod::shape(seq)),
                            )),
                        );
                        // the product is the lower triangle only; the library's own `reconstruct`
                        // then makes it self-adjoint, so the check does the same
                        for j in 1..s.n {
                            for i in 0..j {
                                out[(i, j)] = T::conj_impl(&out[(j, i)]);
                            }
                        }
                        let recon = to_layout(out.as_ref(), s.n, s.n, s.order);
                        within::<T>(max_diff(a0, &recon), frob(a0), s.n, "cholesky")?;
                    }
                    Ok(())
                },
            ))
        }

        // ----------------------------------------------------------- solve

        fn solve_op<T: Scalar + ComplexField>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
            let a0 = diag_dominant::<T>(n, batch, order, seed(n, n, 13));
            let b0 = fill_seeded::<T>(n * batch, seed(n, 1, 17));
            struct S<T> {
                a0: Vec<T>,
                b0: Vec<T>,
                b: Vec<T>,
                sink: T,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                b0: b0.clone(),
                b: b0.clone(),
                sink: T::zero(),
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.b.copy_from_slice(&s.b0);
                    Ok(())
                },
                |s, mut tk| {
                    for bi in 0..s.batch {
                        let a = mref(&s.a0[bi * s.n * s.n..], s.n, s.n, s.order);
                        let (l, u, fwd, bwd) = partial_lu_of(a, &mut tk, s.par);
                        let rhs = mmut(&mut s.b[bi * s.n..], s.n, 1, s.order);
                        plu_solve::solve_in_place_with_conj(
                            l.as_ref(),
                            u.as_ref(),
                            PermRef::new_checked(&fwd, &bwd, s.n),
                            Conj::No,
                            rhs,
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(
                                plu_solve::solve_in_place_scratch::<usize, T>(s.n, 1, parmod::shape(s.par)),
                            )),
                        );
                        acc(&mut s.sink, s.b[bi * s.n].clone());
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

        // -------------------------------------------------- triangular_solve

        fn triangular_solve_op<T: Scalar + ComplexField>(
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
                sink: T,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                u0,
                b0: b0.clone(),
                b: b0.clone(),
                sink: T::zero(),
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.b.copy_from_slice(&s.b0);
                    Ok(())
                },
                |s, mut tk| {
                    for bi in 0..s.batch {
                        let u = mref(&s.u0[bi * s.n * s.n..], s.n, s.n, s.order);
                        let rhs = mmut(&mut s.b[bi * s.n..], s.n, 1, s.order);
                        fk::linalg::triangular_solve::solve_upper_triangular_in_place(u, rhs, parmod::mk_from(&mut tk, s.par));
                        acc(&mut s.sink, s.b[bi * s.n].clone());
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

        // ----------------------------------------------------- svd_values

        fn svd_values_op<T: Scalar + ComplexField>(
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
        ) -> Result<Op, String>
        where
            T::Real: Scalar,
        {
            let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 29));
            struct S<T: ComplexField> {
                a0: Vec<T>,
                outputs: Vec<Vec<T::Real>>,
                stash: Vec<Vec<T::Real>>,
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                outputs: Vec::with_capacity(batch),
                stash: Vec::new(),
                m,
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.stash.append(&mut s.outputs);
                    Ok(())
                },
                |s, mut tk| {
                    s.stash.clear(); // the previous call's outputs are freed inside the clock
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        // `MatRef::singular_values`, with the row's parallelism.
                        let a = mref(a0, s.m, s.n, s.order);
                        let mut sv = fk::diag::Diag::<T>::zeros(s.m.min(s.n));
                        let req = fk::linalg::svd::svd_scratch::<T>(
                            s.m,
                            s.n,
                            fk::linalg::svd::ComputeSvdVectors::No,
                            fk::linalg::svd::ComputeSvdVectors::No,
                            parmod::shape(s.par),
                            Default::default(),
                        );
                        fk::linalg::svd::svd(
                            a,
                            sv.as_mut(),
                            None,
                            None,
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(req)),
                            Default::default(),
                        )
                        .map_err(|e| format!("singular_values failed: {e:?}"))?;
                        s.outputs.push(sv.column_vector().iter().map(|x| T::real_part_impl(x)).collect());
                    }
                    Ok(())
                },
                |s| {
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        let sv = &s.outputs[bi];
                        check_singular_values::<T>(a0, sv, s.m, s.n)?;
                    }
                    Ok(())
                },
            ))
        }

        // ------------------------------------------------------- svd_thin

        fn svd_thin_op<T: Scalar + ComplexField>(
            m: usize,
            n: usize,
            batch: usize,
            order: Order,
        ) -> Result<Op, String> {
            let a0 = fill_seeded::<T>(m * n * batch, seed(m, n, 31));
            struct S<T: ComplexField> {
                a0: Vec<T>,
                outputs: Vec<(fk::diag::Diag<T>, fk::Mat<T>, fk::Mat<T>)>,
                stash: Vec<(fk::diag::Diag<T>, fk::Mat<T>, fk::Mat<T>)>,
                m: usize,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                outputs: Vec::with_capacity(batch),
                stash: Vec::new(),
                m,
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.stash.append(&mut s.outputs);
                    Ok(())
                },
                |s, mut tk| {
                    s.stash.clear(); // the previous call's outputs are freed inside the clock
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        // `MatRef::thin_svd`, with the row's parallelism.
                        let a = mref(a0, s.m, s.n, s.order);
                        let r = s.m.min(s.n);
                        let mut sv = fk::diag::Diag::<T>::zeros(r);
                        let mut u = fk::Mat::<T>::zeros(s.m, r);
                        let mut v = fk::Mat::<T>::zeros(s.n, r);
                        let req = fk::linalg::svd::svd_scratch::<T>(
                            s.m,
                            s.n,
                            fk::linalg::svd::ComputeSvdVectors::Thin,
                            fk::linalg::svd::ComputeSvdVectors::Thin,
                            parmod::shape(s.par),
                            Default::default(),
                        );
                        fk::linalg::svd::svd(
                            a,
                            sv.as_mut(),
                            Some(u.as_mut()),
                            Some(v.as_mut()),
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(req)),
                            Default::default(),
                        )
                        .map_err(|e| format!("thin_svd failed: {e:?}"))?;
                        s.outputs.push((sv, u, v));
                    }
                    Ok(())
                },
                |s| {
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.m * s.n..(bi + 1) * s.m * s.n];
                        let (sv, uu, vv) = &s.outputs[bi];
                        let r = sv.dim();
                        let u = to_layout(uu.as_ref(), s.m, r, s.order);
                        let v = to_layout(vv.as_ref(), s.n, r, s.order);
                        let scale = frob(a0);
                        within::<T>(gram_residual(&u, s.m, r, s.order), 1.0, s.m, "svd_thin U")?;
                        within::<T>(gram_residual(&v, s.n, r, s.order), 1.0, s.n, "svd_thin V")?;
                        let s_diag = diag_vec(sv.as_ref());
                        let scaled_v = scale_cols(&v, s.n, r, &s_diag, s.order);
                        let mut recon = vec![T::zero(); s.m * s.n];
                        mul_adj_b(s.m, r, s.n, s.order, &u, &scaled_v, &mut recon);
                        within::<T>(max_diff(a0, &recon), scale, s.n.max(s.m), "svd_thin")?;
                    }
                    Ok(())
                },
            ))
        }

        // ---------------------------------------------------------- eigh

        fn eigh_op<T: Scalar + ComplexField>(n: usize, batch: usize, order: Order) -> Result<Op, String> {
            let a0 = hermitian::<T>(n, order, seed(n, n, 37));
            let a0 = replicate(&a0, batch);
            struct S<T: ComplexField> {
                a0: Vec<T>,
                outputs: Vec<(fk::diag::Diag<T>, fk::Mat<T>)>,
                stash: Vec<(fk::diag::Diag<T>, fk::Mat<T>)>,
                n: usize,
                batch: usize,
                order: Order,
                par: ParCfg,
            }
            let state = S {
                a0,
                outputs: Vec::with_capacity(batch),
                stash: Vec::new(),
                n,
                batch,
                order,
                par: cfg_par(),
            };
            Ok(op_from(
                state,
                String::new(),
                |s| {
                    s.stash.append(&mut s.outputs);
                    Ok(())
                },
                |s, mut tk| {
                    s.stash.clear(); // the previous call's outputs are freed inside the clock
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                        // `MatRef::self_adjoint_eigen(Side::Lower)`, with the row's parallelism.
                        let a = mref(a0, s.n, s.n, s.order);
                        let mut u = fk::Mat::<T>::zeros(s.n, s.n);
                        let mut sv = fk::diag::Diag::<T>::zeros(s.n);
                        let req = fk::linalg::evd::self_adjoint_evd_scratch::<T>(
                            s.n,
                            fk::linalg::evd::ComputeEigenvectors::Yes,
                            parmod::shape(s.par),
                            Default::default(),
                        );
                        fk::linalg::evd::self_adjoint_evd(
                            a,
                            sv.as_mut(),
                            Some(u.as_mut()),
                            parmod::mk_from(&mut tk, s.par),
                            MemStack::new(&mut MemBuffer::new(req)),
                            Default::default(),
                        )
                        .map_err(|e| format!("self_adjoint_eigen failed: {e:?}"))?;
                        s.outputs.push((sv, u));
                    }
                    Ok(())
                },
                |s| {
                    for bi in 0..s.batch {
                        let a0 = &s.a0[bi * s.n * s.n..(bi + 1) * s.n * s.n];
                        let (sv, uu) = &s.outputs[bi];
                        let u = to_layout(uu.as_ref(), s.n, s.n, s.order);
                        let scale = frob(a0);
                        within::<T>(gram_residual(&u, s.n, s.n, s.order), 1.0, s.n, "eigh U")?;
                        let s_diag = diag_vec(sv.as_ref());
                        let mut au = vec![T::zero(); s.n * s.n];
                        mul(s.n, s.n, s.n, s.order, a0, &u, &mut au);
                        let us = scale_cols(&u, s.n, s.n, &s_diag, s.order);
                        within::<T>(max_diff(&au, &us), scale, s.n, "eigh")?;
                    }
                    Ok(())
                },
            ))
        }
    };
}

define_backend!(fork, faer_fork, fork_par, explicit_ops);
define_backend!(upstream, faer_upstream, upstream_par, highlevel_ops);
