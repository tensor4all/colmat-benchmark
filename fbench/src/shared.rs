//! Types and provider-independent reference math shared by both faer arms and
//! the vendor arm.
//!
//! Everything here is written in terms of plain slices and the `Scalar` trait, so
//! the same code checks a fork result, an upstream result and a LAPACK result.
//! Correctness is always a residual (or a norm identity) computed with this
//! reference math, never a comparison between two providers.

use num_complex::{Complex32, Complex64};

/// The scalar types the harness measures.
///
/// All four are faer's own native dtypes. The vendor arm has a route for `gemm` in
/// every one of them and for the LAPACK drivers in `f64`/`c64` only, which the
/// vendor adapter reports as a skip rather than a number.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F64,
    C64,
    F32,
    C32,
}

impl Dtype {
    /// The dtype label used in the CSV.
    pub fn label(self) -> &'static str {
        match self {
            Dtype::F64 => "f64",
            Dtype::C64 => "c64",
            Dtype::F32 => "f32",
            Dtype::C32 => "c32",
        }
    }
}

/// The memory-order axis: `C` (row-major) or `F` (column-major).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Order {
    C,
    F,
}

impl Order {
    pub fn label(self) -> &'static str {
        match self {
            Order::C => "c",
            Order::F => "f",
        }
    }
}

/// One requested case: `m x n` (and `k` for `gemm`), `batch` matrices.
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub m: usize,
    pub n: usize,
    pub k: Option<usize>,
    pub batch: usize,
}

impl Case {
    /// The shape string the suite declares: `MxKxN`, `MxN` or `N`.
    pub fn shape(&self) -> String {
        match self.k {
            Some(k) => format!("{}x{}x{}", self.m, k, self.n),
            None if self.m == self.n => self.n.to_string(),
            None => format!("{}x{}", self.m, self.n),
        }
    }
}

/// A scalar the reference math and both providers support.
pub trait Scalar: Copy + Send + Sync + 'static {
    const IS_COMPLEX: bool;
    /// The residual floor of this dtype, relative to the case's scale.
    ///
    /// `f64`'s value is the one the campaign was recorded with and is scaled by the case
    /// size. The `f32` pair is not: at f32 an `n`-scaled floor would admit a percent-level
    /// error in a spectrum at the tall sizes the suite declares, so it is flat. Its number is
    /// the observed residual, not a proof: `svd_thin`'s reconstruction reaches 92 eps at n=96
    /// and grows like `sqrt(n)`, so the floor is 500 eps of binary32 and stays there.
    const TOL: f64;
    /// Whether the floor is multiplied by the case size.
    const TOL_SCALES_WITH_N: bool;
    fn zero() -> Self;
    fn one() -> Self;
    fn from_f64(x: f64) -> Self;
    fn from_parts(re: f64, im: f64) -> Self;
    fn add(self, other: Self) -> Self;
    fn sub(self, other: Self) -> Self;
    fn mul(self, other: Self) -> Self;
    fn add_assign(&mut self, other: Self);
    fn conj(self) -> Self;
    /// The modulus.
    fn abs(self) -> f64;
}

impl Scalar for f64 {
    const IS_COMPLEX: bool = false;
    const TOL: f64 = 1e-10;
    const TOL_SCALES_WITH_N: bool = true;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(x: f64) -> Self {
        x
    }
    fn from_parts(re: f64, _im: f64) -> Self {
        re
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }
    fn conj(self) -> Self {
        self
    }
    fn abs(self) -> f64 {
        f64::abs(self)
    }
}

impl Scalar for Complex64 {
    const IS_COMPLEX: bool = true;
    const TOL: f64 = 1e-9;
    const TOL_SCALES_WITH_N: bool = true;
    fn zero() -> Self {
        Complex64::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex64::new(1.0, 0.0)
    }
    fn from_f64(x: f64) -> Self {
        Complex64::new(x, 0.0)
    }
    fn from_parts(re: f64, im: f64) -> Self {
        Complex64::new(re, im)
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }
    fn conj(self) -> Self {
        Complex64::conj(&self)
    }
    fn abs(self) -> f64 {
        Complex64::norm(self)
    }
}

impl Scalar for f32 {
    const IS_COMPLEX: bool = false;
    const TOL: f64 = 6e-5;
    const TOL_SCALES_WITH_N: bool = false;
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn from_parts(re: f64, _im: f64) -> Self {
        re as f32
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }
    fn conj(self) -> Self {
        self
    }
    fn abs(self) -> f64 {
        f64::from(f32::abs(self))
    }
}

impl Scalar for Complex32 {
    const IS_COMPLEX: bool = true;
    const TOL: f64 = 6e-5;
    const TOL_SCALES_WITH_N: bool = false;
    fn zero() -> Self {
        Complex32::new(0.0, 0.0)
    }
    fn one() -> Self {
        Complex32::new(1.0, 0.0)
    }
    fn from_f64(x: f64) -> Self {
        Complex32::new(x as f32, 0.0)
    }
    fn from_parts(re: f64, im: f64) -> Self {
        Complex32::new(re as f32, im as f32)
    }
    fn add(self, other: Self) -> Self {
        self + other
    }
    fn sub(self, other: Self) -> Self {
        self - other
    }
    fn mul(self, other: Self) -> Self {
        self * other
    }
    fn add_assign(&mut self, other: Self) {
        *self += other;
    }
    fn conj(self) -> Self {
        Complex32::conj(&self)
    }
    fn abs(self) -> f64 {
        f64::from(Complex32::norm(self))
    }
}

/// The linear index of `(i, j)` in an `m x n` matrix stored in `order`.
#[inline]
pub fn idx(i: usize, j: usize, m: usize, n: usize, order: Order) -> usize {
    match order {
        Order::C => i * n + j,
        Order::F => j * m + i,
    }
}

/// The tolerance for a case of size `n`: a small multiple of the dtype floor.
pub fn tol_of<T: Scalar>(n: usize) -> f64 {
    if T::TOL_SCALES_WITH_N {
        T::TOL * (n.max(1) as f64)
    } else {
        T::TOL
    }
}

/// `*dst += value`, spelled through [`Scalar`] so it is not ambiguous with the
/// `core::ops::AddAssign` impl the concrete scalar types also carry.
#[inline]
pub fn acc<T: Scalar>(dst: &mut T, value: T) {
    <T as Scalar>::add_assign(dst, value);
}

/// One case seed formula for all providers. Callers must also agree on the salt.
pub fn seed(m: usize, n: usize, salt: usize) -> u64 {
    (m as u64)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add((n as u64).wrapping_mul(0x85EB_CA6B))
        .wrapping_add((salt as u64).wrapping_mul(0xC2B2_AE35))
        .wrapping_add(0x5EED)
}

/// Deterministic pseudo-random values in `[-1, 1]` (complex: both parts),
/// seeded from the case so `verify` and `run` see the same matrices.
pub fn fill_seeded<T: Scalar>(len: usize, seed: u64) -> Vec<T> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut unit = || (next() >> 11) as f64 / ((1u64 << 53) as f64);
    (0..len)
        .map(|_| {
            let re = 2.0 * unit() - 1.0;
            if T::IS_COMPLEX {
                T::from_parts(re, 2.0 * unit() - 1.0)
            } else {
                T::from_f64(re)
            }
        })
        .collect()
}

/// A diagonally dominant upper triangular `n x n` matrix, replicated `batch` times: the
/// `triangular_solve` input. One definition for every arm (the faer rows and the vendor's `?trsm`),
/// so every row of a cell solves the same system.
pub fn upper_triangular<T: Scalar>(n: usize, batch: usize, order: Order, seed: u64) -> Vec<T> {
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
    let mut replicated = Vec::with_capacity(out.len() * batch);
    for _ in 0..batch {
        replicated.extend_from_slice(&out);
    }
    replicated
}

/// Blocking of [`mm`], in elements. `JBLK` columns of `c` are accumulated at once, so `a`'s block is
/// streamed once per `JBLK` columns rather than once per column; `IBLK` rows of `c` and `KBLK` terms
/// of `p` are what stay resident in L1/L2 while that stream runs.
const IBLK: usize = 256;
const JBLK: usize = 8;
const KBLK: usize = 256;

/// `c = a * b` for column-major `a` (`l x k`), `b` (`k x n`) and `c` (`l x n`).
///
/// Cache-blocked, and only *tiled*: `c[i, j]` is still the sum over `p` in ascending order starting
/// from zero, so every element is bit-identical to the textbook triple loop (kept as the test
/// reference, `tests::naive_mul`) and no residual can move because the product got faster.
fn mm<T: Scalar>(l: usize, k: usize, n: usize, a: &[T], b: &[T], c: &mut [T]) {
    use rayon::prelude::*;
    c.fill(T::zero());
    if l == 0 || n == 0 {
        return;
    }
    // The `JBLK`-column blocks of `c` are disjoint slices, so they run on the rayon pool: the
    // pool is the thread budget of the process (`configure_threads`), and the check runs outside
    // every clocked region. Each element of `c` is still the sum over `p` in ascending order, so
    // the result is bit-identical to the sequential loop; only the wall time changes (a check at
    // 16384 was hours on one core, issue #97).
    c.par_chunks_mut(l * JBLK).enumerate().for_each(|(block, cblock)| {
        let jb = block * JBLK;
        let jw = cblock.len() / l;
        let mut ib = 0;
        while ib < l {
            let iw = IBLK.min(l - ib);
            let mut pb = 0;
            while pb < k {
                let pw = KBLK.min(k - pb);
                for p in pb..pb + pw {
                    let acol = &a[p * l + ib..][..iw];
                    for jj in 0..jw {
                        let bv = b[p + (jb + jj) * k];
                        let ccol = &mut cblock[jj * l + ib..][..iw];
                        for i in 0..iw {
                            ccol[i].add_assign(acol[i].mul(bv));
                        }
                    }
                }
                pb += pw;
            }
            ib += iw;
        }
    });
}

/// `c = a * b` with `a` `m x k` and `b` `k x n`, all in the same `order`.
pub fn mul<T: Scalar>(m: usize, k: usize, n: usize, order: Order, a: &[T], b: &[T], c: &mut [T]) {
    // A row-major buffer is the column-major transpose of the same matrix, and a product's transpose
    // is the product of the transposes in reverse order (`(a b)^T = b^T a^T`), so the row-major route
    // is [`mm`] on the same two buffers with the operands swapped: nothing is copied, and the only
    // difference is that each element is formed as `b * a` rather than `a * b`, which is the same
    // IEEE products in the same order for both real and complex scalars.
    match order {
        Order::F => mm(m, k, n, a, b, c),
        Order::C => mm(n, k, m, b, a, c),
    }
}

/// `c = a * b^H` with `a` `m x k` and `b` `n x k`.
pub fn mul_adj_b<T: Scalar>(
    m: usize,
    k: usize,
    n: usize,
    order: Order,
    a: &[T],
    b: &[T],
    c: &mut [T],
) {
    // The adjoint side is the one operand that has to be materialised, in the layout the kernel
    // walks. Column-major that is `b^H` (`k x n`) and `mm` runs as it stands; row-major the kernel
    // computes `c^T = (a b^H)^T = conj(b) a^T`, so it is `conj(b)` (`n x k`) with the operands
    // swapped, exactly as in `mul`. Either way the pass is O(nk) against a product that is O(nkm).
    match order {
        Order::F => {
            let mut bh = vec![T::zero(); k * n];
            for j in 0..n {
                for p in 0..k {
                    bh[p + j * k] = b[j + p * n].conj();
                }
            }
            mm(m, k, n, a, &bh, c);
        }
        Order::C => {
            let mut cb = vec![T::zero(); n * k];
            for j in 0..n {
                for p in 0..k {
                    cb[j + p * n] = b[j * k + p].conj();
                }
            }
            mm(n, k, m, &cb, a, c);
        }
    }
}

/// `src^H` (`cols x rows`) of an `rows x cols` `src`, in the same `order`.
fn adjoint<T: Scalar>(src: &[T], rows: usize, cols: usize, order: Order) -> Vec<T> {
    let mut out = vec![T::zero(); cols * rows];
    for i in 0..cols {
        for j in 0..rows {
            out[idx(i, j, cols, rows, order)] = src[idx(j, i, rows, cols, order)].conj();
        }
    }
    out
}

/// The largest modulus of `q^H q - I` for an `m x n` `q`, an orthonormality check.
pub fn gram_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
    // `q^H q` is `q^H (q^H)^H`, i.e. `mul_adj_b` on the adjoint of `q`.
    let qh = adjoint(q, m, n, order);
    let mut gram = vec![T::zero(); n * n];
    mul_adj_b(n, m, n, order, &qh, &qh, &mut gram);
    gram_identity_residual(&gram, n, order)
}

/// The largest modulus of `q q^H - I` for an `m x n` `q`, the row-side
/// orthonormality check.
pub fn gram_adj_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
    let mut gram = vec![T::zero(); m * m];
    mul_adj_b(m, n, m, order, q, q, &mut gram);
    gram_identity_residual(&gram, m, order)
}

/// `max |g - I|` over an `n x n` Gram matrix, as the two checks above report it.
fn gram_identity_residual<T: Scalar>(g: &[T], n: usize, order: Order) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { T::one() } else { T::zero() };
            worst = residual_max(worst, g[idx(i, j, n, n, order)].sub(target).abs());
        }
    }
    worst
}

fn residual_max(a: f64, b: f64) -> f64 {
    // f64::max ignores NaN, which would turn a corrupt result into a zero residual.
    if a.is_finite() && b.is_finite() {
        a.max(b)
    } else {
        f64::INFINITY
    }
}

/// `max |a_i - b_i|` over two buffers in the same layout.
pub fn max_diff<T: Scalar>(a: &[T], b: &[T]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| x.sub(*y).abs())
        .fold(0.0_f64, residual_max)
}

/// The Frobenius norm of a buffer.
pub fn frob<T: Scalar>(a: &[T]) -> f64 {
    a.iter().map(|x| x.abs() * x.abs()).sum::<f64>().sqrt()
}

/// A Hermitian `n x n` matrix, `(M + M^H) / 2`.
pub fn hermitian<T: Scalar>(n: usize, order: Order, seed: u64) -> Vec<T> {
    let m: Vec<T> = fill_seeded(n * n, seed);
    let half = T::from_f64(0.5);
    let mut out = vec![T::zero(); n * n];
    for j in 0..n {
        for i in 0..n {
            out[idx(i, j, n, n, order)] = m[idx(i, j, n, n, order)]
                .add(m[idx(j, i, n, n, order)].conj())
                .mul(half);
        }
    }
    out
}

/// Scale each column `j` of `u` (m x r) by `s[j]`, producing a buffer in `order`.
pub fn scale_cols<T: Scalar>(u: &[T], m: usize, r: usize, s: &[T], order: Order) -> Vec<T> {
    let mut out = vec![T::zero(); m * r];
    for j in 0..r {
        for i in 0..m {
            out[idx(i, j, m, r, order)] = u[idx(i, j, m, r, order)].mul(s[j]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn residuals_reject_missing_and_nonfinite_values() {
        assert!(!max_diff(&[f64::NAN], &[1.0]).is_finite());
        assert!(!max_diff(&[1.0, 2.0], &[1.0]).is_finite());
        assert!(!gram_residual(&[f64::NAN], 1, 1, Order::C).is_finite());
        assert!(!gram_adj_residual(&[f64::NAN], 1, 1, Order::F).is_finite());
    }

    #[test]
    fn reference_matmul_is_layout_independent() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let mut c = [0.0; 4];
        mul(2, 3, 2, Order::C, &a, &b, &mut c);
        assert_eq!(c, [58.0, 64.0, 139.0, 154.0]);
        let a_f = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let b_f = [7.0, 9.0, 11.0, 8.0, 10.0, 12.0];
        let mut d = [0.0; 4];
        mul(2, 3, 2, Order::F, &a_f, &b_f, &mut d);
        assert_eq!(d, [58.0, 139.0, 64.0, 154.0]);
    }

    /// The O(n^3) triple loop `mul` used to be, for both layouts: the reference the blocked kernel
    /// is checked against, and the only place in this crate the naive product still exists (issue
    /// #77). It is private to this test module, so a check path cannot name it in a release *or* a
    /// test build; `check_paths_do_not_call_the_naive_product` catches a re-added copy.
    fn naive_mul<T: Scalar>(
        m: usize,
        k: usize,
        n: usize,
        order: Order,
        a: &[T],
        b: &[T],
    ) -> Vec<T> {
        let mut c = vec![T::zero(); m * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = T::zero();
                for p in 0..k {
                    acc.add_assign(a[idx(i, p, m, k, order)].mul(b[idx(p, j, k, n, order)]));
                }
                c[idx(i, j, m, n, order)] = acc;
            }
        }
        c
    }

    fn naive_mul_adj_b<T: Scalar>(
        m: usize,
        k: usize,
        n: usize,
        order: Order,
        a: &[T],
        b: &[T],
    ) -> Vec<T> {
        let mut c = vec![T::zero(); m * n];
        for j in 0..n {
            for i in 0..m {
                let mut acc = T::zero();
                for p in 0..k {
                    acc.add_assign(a[idx(i, p, m, k, order)].mul(b[idx(j, p, n, k, order)].conj()));
                }
                c[idx(i, j, m, n, order)] = acc;
            }
        }
        c
    }

    fn naive_gram_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let mut acc = T::zero();
                for p in 0..m {
                    acc.add_assign(
                        q[idx(p, i, m, n, order)]
                            .conj()
                            .mul(q[idx(p, j, m, n, order)]),
                    );
                }
                let target = if i == j { T::one() } else { T::zero() };
                worst = residual_max(worst, acc.sub(target).abs());
            }
        }
        worst
    }

    fn naive_gram_adj_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
        let mut worst = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let mut acc = T::zero();
                for p in 0..n {
                    acc.add_assign(q[idx(i, p, m, n, order)].mul(q[idx(j, p, m, n, order)].conj()));
                }
                let target = if i == j { T::one() } else { T::zero() };
                worst = residual_max(worst, acc.sub(target).abs());
            }
        }
        worst
    }

    /// **The blocked kernel is the triple loop, bit for bit.** The check products got faster by
    /// tiling only -- every element is still the sum over `p` in ascending order from zero -- so a
    /// residual (and therefore every `verify` verdict and every published number that depends on a
    /// fixture built through these functions) cannot move. The shapes cross the `IBLK` row seam
    /// (300 > 256), the `JBLK` column seam (17, 71, 129, 300 > 8) and, with `k = 300`, the `KBLK`
    /// seam on the `p` axis. The `spd` fixture the `cholesky` and `solve` cases are built from is not
    /// called here, but it goes through the same `mul_adj_b` this pins.
    fn blocked_products_match_the_textbook_loop<T: Scalar>() {
        for (m, k, n) in [
            (1, 1, 1),
            (3, 5, 2),
            (64, 64, 64),
            (129, 37, 71),
            (40, 300, 17),
            (300, 4, 2),
        ] {
            for order in [Order::C, Order::F] {
                let a = fill_seeded::<T>(m * k, seed(m, k, 1));
                let b = fill_seeded::<T>(k * n, seed(k, n, 2));
                let mut c = vec![T::zero(); m * n];
                mul(m, k, n, order, &a, &b, &mut c);
                assert_eq!(
                    max_diff(&c, &naive_mul(m, k, n, order, &a, &b)),
                    0.0,
                    "mul {m}x{k}x{n} {}",
                    order.label()
                );
                let bh = fill_seeded::<T>(n * k, seed(n, k, 3));
                mul_adj_b(m, k, n, order, &a, &bh, &mut c);
                assert_eq!(
                    max_diff(&c, &naive_mul_adj_b(m, k, n, order, &a, &bh)),
                    0.0,
                    "mul_adj_b {m}x{k}x{n} {}",
                    order.label()
                );
            }
        }
        // the two orthonormality products, at a size where the triple loop is still cheap
        for (m, n) in [(1, 1), (5, 3), (37, 13), (13, 37)] {
            for order in [Order::C, Order::F] {
                let q = fill_seeded::<T>(m * n, seed(m, n, 4));
                assert_eq!(
                    gram_residual(&q, m, n, order),
                    naive_gram_residual(&q, m, n, order),
                    "gram_residual {m}x{n} {}",
                    order.label()
                );
                assert_eq!(
                    gram_adj_residual(&q, m, n, order),
                    naive_gram_adj_residual(&q, m, n, order),
                    "gram_adj_residual {m}x{n} {}",
                    order.label()
                );
            }
        }
    }

    #[test]
    fn blocked_products_match_the_textbook_loop_all_dtypes() {
        blocked_products_match_the_textbook_loop::<f64>();
        blocked_products_match_the_textbook_loop::<Complex64>();
        blocked_products_match_the_textbook_loop::<f32>();
        blocked_products_match_the_textbook_loop::<Complex32>();
    }

    /// Issue #77: no check path may spend minutes on a naive product. Every product in a `check`
    /// closure goes through [`mul`]/[`mul_adj_b`]/[`gram_residual`]/[`gram_adj_residual`] here; the
    /// triple loop is `tests::naive_mul`, and this scan fails if a copy of it comes back.
    #[test]
    fn check_paths_do_not_call_the_naive_product() {
        for (name, src) in [
            ("backend.rs", include_str!("backend.rs")),
            ("lapack.rs", include_str!("lapack.rs")),
            ("vendor.rs", include_str!("vendor.rs")),
            ("main.rs", include_str!("main.rs")),
            ("timing.rs", include_str!("timing.rs")),
            ("op.rs", include_str!("op.rs")),
        ] {
            assert!(
                !src.contains("naive_mul"),
                "{name} calls the naive O(n^3) product in a check path (issue #77): use \
                 shared::mul/mul_adj_b/gram_*, which are blocked"
            );
        }
    }
}
