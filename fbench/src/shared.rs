//! Types and provider-independent reference math shared by both faer arms and
//! the vendor arm.
//!
//! Everything here is written in terms of plain slices and the `Scalar` trait, so
//! the same code checks a fork result, an upstream result and a LAPACK result.
//! Correctness is always a residual (or a norm identity) computed with this
//! reference math, never a comparison between two providers.
//!
//! The loops here are serial and cubic in the case size, and they run inside a cell's
//! check. They poll the cell clock (`timing::expired`), because this is where a cell
//! that cannot finish spends its time: 27.6 h of one weekend campaign went into one
//! `gram_residual` at `n = 4096`.

use num_complex::{Complex32, Complex64};

use crate::timing::expired;

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
    /// The real and imaginary parts, for the reference math's inner loops: they accumulate an
    /// `f64` pair, so the loop vectorizes where `T`'s own arithmetic (a `Complex` multiply through
    /// two struct fields) would not.
    fn parts(self) -> (f64, f64);
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
    fn parts(self) -> (f64, f64) {
        (self, 0.0)
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
    fn parts(self) -> (f64, f64) {
        (self.re, self.im)
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
    fn parts(self) -> (f64, f64) {
        (f64::from(self), 0.0)
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
    fn parts(self) -> (f64, f64) {
        (f64::from(self.re), f64::from(self.im))
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
            // Cut short by the cell limit: the buffer is then only partly written, which the
            // caller's comparison rejects. The executor records the cell as `timeout`, not as a
            // mismatch, so the partial product is never reported as a wrong answer.
            if expired() {
                return;
            }
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
    // Row-major: `c`'s row and `b`'s row are both contiguous, so stream them together -- the
    // mirror of the column-major form above. Each `c[i, j]` is still the sum over `p` in ascending
    // order starting from zero, so the arithmetic is unchanged; a dot-product form (the layout's
    // `idx` walk with the branch hoisted out) measured no better than the old code, because it
    // re-reads `b` with stride `n` for every `i`.
    for i in 0..m {
        if expired() {
            return;
        }
        let arow = &a[i * k..(i + 1) * k];
        let crow = &mut c[i * n..(i + 1) * n];
        crow.fill(T::zero());
        for p in 0..k {
            let aip = arow[p];
            let brow = &b[p * n..(p + 1) * n];
            for j in 0..n {
                crow[j].add_assign(aip.mul(brow[j]));
            }
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
            if expired() {
                return;
            }
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
    // Row-major: `b`'s `p`-th column is strided, so it is transposed once into a column-major
    // scratch (`k*n` elements against `m*n*k` multiply-adds) and then streamed, as `mul` does.
    // `b` is `n x k` (row `j` is `b[j, p]`); transposed it is `p*n + j`, which `transpose` gives
    // for the shape it takes -- the input's own `n x k`, not the product's.
    let bt = transpose(b, n, k, Order::C);
    for i in 0..m {
        if expired() {
            return;
        }
        let arow = &a[i * k..(i + 1) * k];
        let crow = &mut c[i * n..(i + 1) * n];
        crow.fill(T::zero());
        for p in 0..k {
            let aip = arow[p];
            let brow = &bt[p * n..(p + 1) * n];
            for j in 0..n {
                crow[j].add_assign(aip.mul(brow[j].conj()));
            }
        }
    }
}

/// The largest modulus of `q^H q - I` for an `m x n` `q`, an orthonormality check.
///
/// Returns an infinite residual if the cell limit is reached: no tolerance accepts it, and the
/// executor turns the resulting failure into a `timeout` row rather than a wrong-answer report.
///
/// The layout is decided once, outside every loop, and the `p` walk is an addition: this inner
/// product runs `n^2 m` times (6.9e10 at `n = 4096`), and the layout test inside it was the
/// reason one such cell took hours. The dot product accumulates `f64` real and imaginary parts, so
/// the loop vectorizes when the layout makes `p` contiguous (column-major). The sum is over `p` in
/// ascending order as before, but an `f32` residual now accumulates in `f64`, so its value is not
/// bit-identical to the old one: the difference is bounded by `f32`'s own eps, and
/// `tests::the_fast_reference_math_agrees_with_the_old_one` fixes it.
pub fn gram_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
    // The kernel wants each column of `q` contiguous. Column-major has that for free; row-major
    // pays an `n*m` copy to get it, which the measured difference justifies: at `n = 1024` the
    // strided form took 2236 ms against 490 ms for the contiguous one (same host, same answer),
    // because every element of a strided column is a cache miss.
    match order {
        Order::F => gram_residual_columns(q, m, n),
        Order::C => gram_residual_columns(&transpose(q, m, n, Order::C), m, n),
    }
}

/// `q` as a column-major matrix of the same logical shape. Polls the cell clock per row, so a cut
/// during the copy stops it; the kernel that follows then returns the cut at its first poll.
fn transpose<T: Scalar>(q: &[T], m: usize, n: usize, from: Order) -> Vec<T> {
    let mut out = vec![T::zero(); m * n];
    for i in 0..n {
        if expired() {
            return out;
        }
        for p in 0..m {
            out[idx(p, i, m, n, if from == Order::C { Order::F } else { Order::C })] =
                q[idx(p, i, m, n, from)];
        }
    }
    out
}

/// Column-major `q`: `(p, i)` is at `i*m + p`, so `p` is contiguous and both columns of the inner
/// product are plain slices.
fn gram_residual_columns<T: Scalar>(q: &[T], m: usize, n: usize) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..n {
        if expired() {
            return f64::INFINITY;
        }
        let column_i = &q[i * m..i * m + m];
        for j in 0..n {
            let column_j = &q[j * m..j * m + m];
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (x, y) in column_i.iter().zip(column_j.iter()) {
                let (xr, xi) = x.parts();
                let (yr, yi) = y.parts();
                re += xr * yr + xi * yi;
                im += xr * yi - xi * yr;
            }
            let (tr, ti) = target(i, j);
            worst = residual_max(worst, T::from_parts(re - tr, im - ti).abs());
        }
    }
    worst
}

/// The identity entry the residual subtracts.
#[inline]
fn target(i: usize, j: usize) -> (f64, f64) {
    if i == j {
        (1.0, 0.0)
    } else {
        (0.0, 0.0)
    }
}

/// The largest modulus of `q q^H - I` for an `m x n` `q`, the row-side
/// orthonormality check. As `gram_residual`, an infinite residual means the cell was cut.
pub fn gram_adj_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
    // As `gram_residual`, with the roles of the layouts swapped: this kernel wants each row of `q`
    // contiguous, which row-major has and column-major buys with an `n*m` copy.
    match order {
        Order::C => gram_adj_residual_rows(q, m, n),
        Order::F => gram_adj_residual_rows(&transpose(q, m, n, Order::F), m, n),
    }
}

/// Row-major `q`: row `i` is contiguous in `p`.
fn gram_adj_residual_rows<T: Scalar>(q: &[T], m: usize, n: usize) -> f64 {
    let mut worst = 0.0_f64;
    for i in 0..m {
        if expired() {
            return f64::INFINITY;
        }
        let row_i = &q[i * n..i * n + n];
        for j in 0..m {
            let row_j = &q[j * n..j * n + n];
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for (x, y) in row_i.iter().zip(row_j.iter()) {
                let (xr, xi) = x.parts();
                let (yr, yi) = y.parts();
                // `q q^H`: `x * conj(y)`
                re += xr * yr + xi * yi;
                im += xi * yr - xr * yi;
            }
            let (tr, ti) = target(i, j);
            worst = residual_max(worst, T::from_parts(re - tr, im - ti).abs());
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

    /// With an expired limit every reference loop returns before writing, so a cut cell cannot
    /// report a partial product as a wrong answer. Delete the `expired()` polls in `mul`,
    /// `mul_adj_b`, `gram_residual` or `gram_adj_residual` and this test fails with an ordinary
    /// value where it expects the cut.
    #[test]
    fn an_expired_limit_stops_the_reference_math_before_it_writes() {
        let m = 8usize;
        let a = vec![1.0f64; m * m];
        let sentinel = -7.0f64;
        crate::timing::arm(Some(core::time::Duration::from_millis(1)));
        std::thread::sleep(core::time::Duration::from_millis(5));

        let mut c = vec![sentinel; m * m];
        mul(m, m, m, Order::F, &a, &a, &mut c);
        assert_eq!(c[0], sentinel, "`mul` must not write after the limit");
        let mut c = vec![sentinel; m * m];
        mul(m, m, m, Order::C, &a, &a, &mut c);
        assert_eq!(
            c[0], sentinel,
            "`mul` (row-major) must not write after the limit"
        );
        let mut c = vec![sentinel; m * m];
        mul_adj_b(m, m, m, Order::F, &a, &a, &mut c);
        assert_eq!(c[0], sentinel, "`mul_adj_b` must not write after the limit");
        assert!(
            gram_residual(&a, m, m, Order::C).is_infinite(),
            "`gram_residual` must report the cut, not a finite residual"
        );
        assert!(
            gram_adj_residual(&a, m, m, Order::F).is_infinite(),
            "`gram_adj_residual` must report the cut"
        );
        crate::timing::disarm();
    }

    // ------------------------------------------------- the old reference math, verbatim
    //
    // These are the pre-hoisting loops, kept as the executable specification the fast versions are
    // compared against. They are not called by the harness: `mul` and `mul_adj_b` here are the
    // single generic form both layouts reduce to, and both gram functions are the `idx`-in-the-
    // inner-loop form that was slow.

    fn old_mul<T: Scalar>(m: usize, k: usize, n: usize, order: Order, a: &[T], b: &[T], c: &mut [T]) {
        if order == Order::F {
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

    fn old_mul_adj_b<T: Scalar>(
        m: usize,
        k: usize,
        n: usize,
        order: Order,
        a: &[T],
        b: &[T],
        c: &mut [T],
    ) {
        if order == Order::F {
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

    fn old_gram_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
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

    fn old_gram_adj_residual<T: Scalar>(q: &[T], m: usize, n: usize, order: Order) -> f64 {
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

    fn close(new: f64, old: f64, tol: f64) -> bool {
        (new - old).abs() <= tol * old.abs().max(1.0)
    }

    /// The fast reference math is the same math: for `f64` and `c64` it agrees with the old loops to
    /// `1e-12` relative. It is **not** bit-identical by contract -- the gram loops accumulate an
    /// `f64` pair instead of `T` -- so this is a numeric bound, not an equality.
    #[test]
    fn the_fast_reference_math_agrees_with_the_old_one() {
        for &(m, k, n) in &[(1usize, 1usize, 1usize), (5, 3, 7), (16, 16, 16), (33, 9, 41)] {
            for order in [Order::C, Order::F] {
                let a = fill_seeded::<f64>(m * k, seed(m, k, 1));
                let b = fill_seeded::<f64>(k * n, seed(k, n, 2));
                let mut want = vec![0.0f64; m * n];
                let mut got = vec![0.0f64; m * n];
                old_mul(m, k, n, order, &a, &b, &mut want);
                mul(m, k, n, order, &a, &b, &mut got);
                for i in 0..m * n {
                    assert!(
                        close(got[i], want[i], 1e-12),
                        "mul f64 {m}x{k}x{n} {}: {} != {}",
                        order.label(),
                        got[i],
                        want[i]
                    );
                }
                let mut want = vec![0.0f64; m * n];
                let mut got = vec![0.0f64; m * n];
                old_mul_adj_b(m, k, n, order, &a, &b, &mut want);
                mul_adj_b(m, k, n, order, &a, &b, &mut got);
                for i in 0..m * n {
                    assert!(
                        close(got[i], want[i], 1e-12),
                        "mul_adj_b f64 {m}x{k}x{n} {}: {} != {}",
                        order.label(),
                        got[i],
                        want[i]
                    );
                }

                let a = fill_seeded::<Complex64>(m * k, seed(m, k, 3));
                let b = fill_seeded::<Complex64>(k * n, seed(k, n, 4));
                let mut want = vec![Complex64::new(0.0, 0.0); m * n];
                let mut got = vec![Complex64::new(0.0, 0.0); m * n];
                old_mul(m, k, n, order, &a, &b, &mut want);
                mul(m, k, n, order, &a, &b, &mut got);
                for i in 0..m * n {
                    assert!(
                        close(got[i].re, want[i].re, 1e-12)
                            && close(got[i].im, want[i].im, 1e-12),
                        "mul c64 {m}x{k}x{n} {}: {got:?} != {want:?}",
                        order.label()
                    );
                }
                let mut want = vec![Complex64::new(0.0, 0.0); m * n];
                let mut got = vec![Complex64::new(0.0, 0.0); m * n];
                old_mul_adj_b(m, k, n, order, &a, &b, &mut want);
                mul_adj_b(m, k, n, order, &a, &b, &mut got);
                for i in 0..m * n {
                    assert!(
                        close(got[i].re, want[i].re, 1e-12)
                            && close(got[i].im, want[i].im, 1e-12),
                        "mul_adj_b c64 {m}x{k}x{n} {}: {got:?} != {want:?}",
                        order.label()
                    );
                }

                // A `q` that is not orthonormal, so both residuals are finite and non-zero.
                let q = fill_seeded::<f64>(m * n, seed(m, n, 5));
                for (got, want) in [
                    (
                        gram_residual(&q, m, n, order),
                        old_gram_residual(&q, m, n, order),
                    ),
                    (
                        gram_adj_residual(&q, m, n, order),
                        old_gram_adj_residual(&q, m, n, order),
                    ),
                ] {
                    assert!(
                        close(got, want, 1e-12),
                        "residual f64 {m}x{n} {}: {got} != {want}",
                        order.label()
                    );
                }
                let q = fill_seeded::<Complex64>(m * n, seed(m, n, 6));
                for (got, want) in [
                    (
                        gram_residual(&q, m, n, order),
                        old_gram_residual(&q, m, n, order),
                    ),
                    (
                        gram_adj_residual(&q, m, n, order),
                        old_gram_adj_residual(&q, m, n, order),
                    ),
                ] {
                    assert!(
                        close(got, want, 1e-12),
                        "residual c64 {m}x{n} {}: {got} != {want}",
                        order.label()
                    );
                }
            }
        }
        // An orthonormal `q` gives a zero residual in both, which is the case a relative bound
        // cannot say anything about.
        let n = 6usize;
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[idx(i, i, n, n, Order::C)] = 1.0;
        }
        for order in [Order::C, Order::F] {
            assert_eq!(gram_residual(&q, n, n, order), 0.0);
            assert_eq!(old_gram_residual(&q, n, n, order), 0.0);
        }
    }

    /// Times the reference math, old and new, at the sizes the campaign meets: the same process, so
    /// the two are measured under the same load. Not part of the suite -- run it with
    /// `cargo test --release --locked reference_math_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn reference_math_cost() {
        for &n in &[128usize, 512, 1024, 2048] {
            let q = fill_seeded::<f64>(n * n, seed(n, n, 11));
            let a = fill_seeded::<f64>(n * n, seed(n, n, 12));
            for order in [Order::C, Order::F] {
                let t = std::time::Instant::now();
                let old = old_gram_residual(&q, n, n, order);
                let old_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = std::time::Instant::now();
                let new = gram_residual(&q, n, n, order);
                let new_ms = t.elapsed().as_secs_f64() * 1e3;
                println!(
                    "n={n} {} gram_residual  old {old_ms:9.1} ms  new {new_ms:9.1} ms  ({old:.3e} / {new:.3e})",
                    order.label()
                );
                let t = std::time::Instant::now();
                let old = old_gram_adj_residual(&q, n, n, order);
                let old_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = std::time::Instant::now();
                let new = gram_adj_residual(&q, n, n, order);
                let new_ms = t.elapsed().as_secs_f64() * 1e3;
                println!(
                    "n={n} {} gram_adj_res  old {old_ms:9.1} ms  new {new_ms:9.1} ms  ({old:.3e} / {new:.3e})",
                    order.label()
                );
                let mut c = vec![0.0f64; n * n];
                let t = std::time::Instant::now();
                old_mul(n, n, n, order, &a, &a, &mut c);
                let old_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = std::time::Instant::now();
                mul(n, n, n, order, &a, &a, &mut c);
                let new_ms = t.elapsed().as_secs_f64() * 1e3;
                println!("n={n} {} mul            old {old_ms:9.1} ms  new {new_ms:9.1} ms", order.label());
                let t = std::time::Instant::now();
                old_mul_adj_b(n, n, n, order, &a, &a, &mut c);
                let old_ms = t.elapsed().as_secs_f64() * 1e3;
                let t = std::time::Instant::now();
                mul_adj_b(n, n, n, order, &a, &a, &mut c);
                let new_ms = t.elapsed().as_secs_f64() * 1e3;
                println!(
                    "n={n} {} mul_adj_b      old {old_ms:9.1} ms  new {new_ms:9.1} ms",
                    order.label()
                );
            }
        }
    }

    /// `f32` accumulates its residual in `f64` now, so it is not the old value: it is closer to the
    /// exact one. The bound is `f32`'s own eps rather than `1e-12`.
    #[test]
    fn the_fast_f32_residual_agrees_to_f32_precision() {
        for &(m, n) in &[(8usize, 8usize), (24, 17)] {
            for order in [Order::C, Order::F] {
                let q = fill_seeded::<f32>(m * n, seed(m, n, 7));
                let want = old_gram_residual(&q, m, n, order);
                let got = gram_residual(&q, m, n, order);
                assert!(
                    close(got, want, 1e-5),
                    "gram_residual f32 {m}x{n} {}: {got} != {want}",
                    order.label()
                );
                let want = old_gram_adj_residual(&q, m, n, order);
                let got = gram_adj_residual(&q, m, n, order);
                assert!(
                    close(got, want, 1e-5),
                    "gram_adj_residual f32 {m}x{n} {}: {got} != {want}",
                    order.label()
                );
            }
        }
    }

    /// The fast loops must keep `residual_max`'s decision about a corrupt result: a `NaN` or an
    /// `inf` anywhere in the input is an infinite residual, never a finite one and never a zero.
    #[test]
    fn the_fast_residuals_still_reject_nonfinite_input() {
        let mut q = fill_seeded::<f64>(4 * 4, seed(4, 4, 8));
        q[5] = f64::NAN;
        for order in [Order::C, Order::F] {
            assert!(gram_residual(&q, 4, 4, order).is_infinite());
            assert!(old_gram_residual(&q, 4, 4, order).is_infinite());
            assert!(gram_adj_residual(&q, 4, 4, order).is_infinite());
            assert!(old_gram_adj_residual(&q, 4, 4, order).is_infinite());
        }
        let mut q = fill_seeded::<Complex64>(4 * 4, seed(4, 4, 9));
        q[7] = Complex64::new(f64::INFINITY, 0.0);
        for order in [Order::C, Order::F] {
            assert!(gram_residual(&q, 4, 4, order).is_infinite());
            assert!(gram_adj_residual(&q, 4, 4, order).is_infinite());
        }
    }

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
