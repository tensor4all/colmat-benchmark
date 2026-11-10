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
pub trait Scalar: Copy + Send + 'static {
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

/// `c = a * b` with `a` `m x k` and `b` `k x n`, all in the same `order`.
pub fn mul<T: Scalar>(m: usize, k: usize, n: usize, order: Order, a: &[T], b: &[T], c: &mut [T]) {
    if order == Order::F {
        // Column-major: accumulate down contiguous columns. Each `c[i, j]` is still the sum over `p` in
        // ascending order starting from zero, so the result is bit-identical to the loop below; only the
        // memory access is cache friendly (the strided loop took minutes at n = 2048).
        for j in 0..n {
            let col = &mut c[j * m..(j + 1) * m];
            col.fill(T::zero());
            for p in 0..k {
                let bpj = b[p + j * k];
                let acol = &a[p * m..(p + 1) * m];
                for i in 0..m {
                    col[i].add_assign(acol[i].mul(bpj));
                }
            }
        }
        return;
    }
    for j in 0..n {
        for i in 0..m {
            let mut acc = T::zero();
            for p in 0..k {
                acc.add_assign(a[idx(i, p, m, k, order)].mul(b[idx(p, j, k, n, order)]));
            }
            c[idx(i, j, m, n, order)] = acc;
        }
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
    if order == Order::F {
        // as in `mul`: same per-element summation order, contiguous access
        for j in 0..n {
            let col = &mut c[j * m..(j + 1) * m];
            col.fill(T::zero());
            for p in 0..k {
                let bjp = b[j + p * n].conj();
                let acol = &a[p * m..(p + 1) * m];
                for i in 0..m {
                    col[i].add_assign(acol[i].mul(bjp));
                }
            }
        }
        return;
    }
    for j in 0..n {
        for i in 0..m {
            let mut acc = T::zero();
            for p in 0..k {
                acc.add_assign(a[idx(i, p, m, k, order)].mul(b[idx(j, p, n, k, order)].conj()));
            }
            c[idx(i, j, m, n, order)] = acc;
        }
    }
}

/// The largest modulus of `q^H q - I` for an `m x n` `q`, an orthonormality check.
pub fn gram_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
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

/// The largest modulus of `q q^H - I` for an `m x n` `q`, the row-side
/// orthonormality check.
pub fn gram_adj_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
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
}
