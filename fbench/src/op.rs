//! The uniform operation a measurement loop drives, independent of provider.
//!
//! A provider arm builds an `Op` for a case out of three closures over its own
//! state: `prepare` restores the caller-visible inputs, `run` performs the
//! measured operation, and `check` verifies the result against that provider's
//! own contract. Keeping the three separate is what lets the timing loop clock
//! only `run`, prepare and restore outside the clock, and still verify the exact
//! state the timed call left behind.

/// The parallelism of the measured call, lent to the loop's `run` closures.
///
/// The measured library's `Par` is not `Copy` from 0.25 on (a team is exclusive authority over
/// reserved members), so a row cannot build one per call from a copyable configuration: the timing
/// loop owns the row's parallelism and lends it. `None` is a row with no team (a budget of one
/// thread, or any provider that is not the fork); the fork's `run` closures hand it a
/// [`Token::Fork`] when the harness was built for the team API (`fork-team`, see
/// `scripts/build_for_faer_rev.sh`).
pub enum Token<'a> {
    /// no team: the row's call is sequential (a budget of one thread, a provider that is not the
    /// fork, or a harness built for a fork without the team API)
    None(core::marker::PhantomData<&'a ()>),
    #[cfg(feature = "fork-team")]
    Fork(colmat::Par<'a>),
}

impl Token<'_> {
    /// The token of a row that carries no team.
    pub fn none() -> Token<'static> {
        Token::None(core::marker::PhantomData)
    }

    pub fn reborrow(&mut self) -> Token<'_> {
        match self {
            Token::None(_) => Token::none(),
            #[cfg(feature = "fork-team")]
            Token::Fork(par) => Token::Fork(par.reborrow()),
        }
    }
}

/// The three steps of an operation over one state, type-erased. `Send` because the timing loop
/// may run inside a held spindle worker group, whose entry requires it; every provider's state is
/// plain buffers.
trait Steps: Send {
    fn prepare(&mut self) -> Result<(), String>;
    fn run(&mut self, token: Token<'_>) -> Result<(), String>;
    fn check(&mut self) -> Result<(), String>;
}

struct Closures<S, P, R, C> {
    state: S,
    prepare: P,
    run: R,
    check: C,
}

impl<S, P, R, C> Steps for Closures<S, P, R, C>
where
    S: Send,
    P: Fn(&mut S) -> Result<(), String> + Send,
    R: for<'a> Fn(&mut S, Token<'a>) -> Result<(), String> + Send,
    C: Fn(&mut S) -> Result<(), String> + Send,
{
    fn prepare(&mut self) -> Result<(), String> {
        (self.prepare)(&mut self.state)
    }
    fn run(&mut self, token: Token<'_>) -> Result<(), String> {
        (self.run)(&mut self.state, token)
    }
    fn check(&mut self) -> Result<(), String> {
        (self.check)(&mut self.state)
    }
}

/// A provider operation for one case.
pub struct Op {
    /// A truthful note recorded in the CSV (`block`, or why a route differs).
    pub note: String,
    steps: Box<dyn Steps>,
}

impl Op {
    /// The same operation with `run` replaced by a no-op: the test that proves every `check`
    /// rejects a result that was never computed.
    #[cfg(test)]
    pub fn without_run(self) -> Op {
        struct NoRun(Box<dyn Steps>);
        impl Steps for NoRun {
            fn prepare(&mut self) -> Result<(), String> {
                self.0.prepare()
            }
            fn run(&mut self, _token: Token<'_>) -> Result<(), String> {
                Ok(())
            }
            fn check(&mut self) -> Result<(), String> {
                self.0.check()
            }
        }
        Op {
            note: self.note,
            steps: Box::new(NoRun(self.steps)),
        }
    }
    pub fn prepare(&mut self) -> Result<(), String> {
        self.steps.prepare()
    }
    pub fn run(&mut self, token: Token<'_>) -> Result<(), String> {
        self.steps.run(token)
    }
    pub fn check(&mut self) -> Result<(), String> {
        self.steps.check()
    }
}

/// Build an [`Op`] whose three closures share one state.
pub fn op_from<S: Send + 'static>(
    state: S,
    note: String,
    prepare: impl Fn(&mut S) -> Result<(), String> + Send + 'static,
    run: impl for<'a> Fn(&mut S, Token<'a>) -> Result<(), String> + Send + 'static,
    check: impl Fn(&mut S) -> Result<(), String> + Send + 'static,
) -> Op {
    Op {
        note,
        steps: Box::new(Closures {
            state,
            prepare,
            run,
            check,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::Token;
    use crate::{
        backend,
        shared::{Case, Dtype, Order},
        vendor,
    };
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::{Mutex, MutexGuard};

    /// A counting allocator, armed only by `no_row_allocates_inside_run`.
    ///
    /// The counter is thread-local and `const`-initialised without a destructor, so reading it
    /// from inside the allocator cannot itself allocate or register a TLS destructor, and a
    /// neighbouring test thread's allocations are not counted.
    ///
    /// What this **cannot** see: an allocation the measured library makes in C. LAPACKE's own
    /// wrappers call `LAPACKE_malloc` -> libc `malloc`
    /// (`netlib-src/source/LAPACKE/src/lapacke_dgesdd.c`), which no Rust allocator observes. The
    /// vendor side of "nothing is allocated inside the clock" is therefore enforced by
    /// construction instead -- the `_work` entry points with a caller-owned workspace, and the
    /// column-major-only route, both in `lapack.rs` -- and this test guards the other half: the
    /// harness itself must never allocate or free on a row's behalf (the `to_vec()`-inside-`run`
    /// class of bug the audit found). docs/reliable-measurement.md states the same limit.
    mod alloc_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static COUNTS: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
        }

        pub struct Counting;

        impl Counting {
            fn add(alloc: bool) {
                let _ = COUNTS.try_with(|c| {
                    let (a, f) = c.get();
                    c.set(if alloc { (a + 1, f) } else { (a, f + 1) });
                });
            }
        }

        /// The `(allocations, frees)` this thread made since the last call.
        pub fn take() -> (usize, usize) {
            COUNTS.with(|c| c.replace((0, 0)))
        }

        // SAFETY: forwarding to `System` with the same layout, counting around it.
        unsafe impl GlobalAlloc for Counting {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                Self::add(true);
                unsafe { System.alloc(layout) }
            }
            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                Self::add(true);
                unsafe { System.alloc_zeroed(layout) }
            }
            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                Self::add(false);
                unsafe { System.dealloc(ptr, layout) }
            }
            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                Self::add(true);
                unsafe { System.realloc(ptr, layout, new_size) }
            }
        }
    }

    #[global_allocator]
    static ALLOC: alloc_probe::Counting = alloc_probe::Counting;

    /// Every family the harness times, shared by the tests below so that a family cannot be
    /// probed by one test and forgotten by another.
    const FAMILIES: &[&str] = &[
        "gemm",
        "qr_factor",
        "qr",
        "lu",
        "full_piv_lu",
        "cholesky",
        "solve",
        "triangular_solve",
        "svd_values",
        "svd_thin",
        "eigh",
        "householder_apply",
    ];

    /// The families whose vendor route is a LAPACK driver, and therefore column-major only
    /// (`lapack.rs`, `DRIVER_FAMILIES`).
    const DRIVER_FAMILIES: &[&str] = &[
        "lu",
        "solve",
        "cholesky",
        "qr_factor",
        "qr",
        "svd_values",
        "svd_thin",
        "eigh",
    ];

    /// The families a faer row times twice: the owning route as the labelled `api-inclusive` row,
    /// and the allocation-free route as the primary (empty note) row.
    const TWO_ROUTE_FAMILIES: &[&str] = &[
        "lu",
        "full_piv_lu",
        "cholesky",
        "solve",
        "qr",
        "svd_values",
        "svd_thin",
        "eigh",
    ];

    /// The thread budget is process-wide (`backend::fork::set_threads`), and the test runner runs
    /// tests on several threads: a test that sets it holds this for its whole body, or another
    /// test's `set_threads(1)` lands between this test's workspace query and its call (a scratch
    /// sized for one thread handed to a two-member team).
    static BUDGET: Mutex<()> = Mutex::new(());

    fn hold_budget() -> MutexGuard<'static, ()> {
        BUDGET
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Every fork family's timed call runs on the token it is lent. A call that builds its own
    /// `Par` instead (`verify_par` is `Par::Seq` under `fork-team`) measures a sequential call at
    /// every thread count, which the fork `lu` and `solve` rows did until this test.
    #[cfg(feature = "fork-team")]
    #[test]
    fn fork_calls_run_on_the_lent_team() {
        use colmat::utils::team::stats;
        let _budget = hold_budget();
        let threads = 2;
        backend::fork::set_threads(threads);
        for family in [
            "gemm",
            "qr_factor",
            "qr",
            "lu",
            "full_piv_lu",
            "cholesky",
            "solve",
            "triangular_solve",
            "svd_values",
            "svd_thin",
            "eigh",
            "householder_apply",
        ] {
            for order in [Order::C, Order::F] {
                // a one-column triangular solve is a matvec, which faer keeps sequential at 256
                let n = if family == "triangular_solve" {
                    2048
                } else {
                    256
                };
                let case = Case {
                    m: n,
                    n,
                    k: if family == "gemm" { Some(n) } else { None },
                    batch: 1,
                };
                for mut op in crate::ops_for("faer-fork", Dtype::F64, family, case, order).unwrap()
                {
                    op.prepare().unwrap();
                    let s = colmat::team::with_team(threads, |par| {
                        stats::enable(true);
                        let _ = stats::take();
                        op.run(Token::Fork(par)).unwrap();
                        stats::enable(false);
                        stats::take()
                    });
                    op.check().unwrap();
                    let team = s.run[2..].iter().sum::<usize>()
                        + s.split[2..].iter().sum::<usize>()
                        + s.gemm_tiled
                        + s.gemm_team_kernel;
                    assert!(
                        team > 0,
                        "faer-fork {family} {} {}: no team dispatch ({s:?})",
                        order.label(),
                        op.note
                    );
                }
            }
        }
        backend::fork::set_threads(1);
    }

    #[test]
    fn checks_reject_a_missing_execution() {
        let _budget = hold_budget();
        backend::fork::set_threads(1);
        backend::upstream::set_threads(1);
        if vendor::NAME != "none" {
            vendor::set_threads(1).unwrap();
        }
        for row in crate::engine_rows() {
            for &family in FAMILIES {
                for dtype in [Dtype::F64, Dtype::C64] {
                    for order in [Order::C, Order::F] {
                        let case = Case {
                            m: 8,
                            n: 8,
                            k: if family == "gemm" { Some(8) } else { None },
                            batch: 2,
                        };
                        // a route the arm does not have (a vendor C-order driver cell, a family
                        // with no LAPACK route) is a skip, never a measurement
                        let ops = match crate::ops_for(&row, dtype, family, case, order) {
                            Ok(ops) => ops,
                            Err(_) => continue,
                        };
                        // householder_apply times both public forms; a faer row times the
                        // two-route families through the owning method and the allocation-free
                        // route.
                        let expected = match family {
                            "householder_apply" => 2,
                            _ if row.starts_with("faer-")
                                && TWO_ROUTE_FAMILIES.contains(&family) =>
                            {
                                2
                            }
                            _ => 1,
                        };
                        assert_eq!(ops.len(), expected);
                        for mut op in ops {
                            op.prepare().unwrap();
                            op.run(crate::op::Token::none()).unwrap();
                            op.check().unwrap();
                            let mut op = op.without_run();
                            op.prepare().unwrap();
                            op.run(crate::op::Token::none()).unwrap();
                            // Missing retained output may fail with Err or bounds panic;
                            // recomputing a fresh correct answer must never pass.
                            let result = catch_unwind(AssertUnwindSafe(|| op.check()));
                            assert!(
                                !matches!(result, Ok(Ok(()))),
                                "{row} {family} {} {} {} checked a replacement result",
                                dtype.label(),
                                order.label(),
                                op.note
                            );
                        }
                    }
                }
            }
        }
    }

    /// **Nothing is allocated or freed inside a timed `run`, on any row.**
    ///
    /// This is the guard for the defect the 2026-11-06 audit found and that the fair-comparison
    /// rule exists to prevent: a `to_vec()`/`Vec` built inside the clock on one arm and prepared
    /// outside it on the other. Every row's `run` is executed single-threaded (`Token::none()` is
    /// `Par::Seq`, `backend.rs`), which is what makes the counter exact: the measured call runs on
    /// this thread and allocates nothing there. A library's own threaded-path allocation is not in
    /// scope here and is declared in `benchmarks/suites/faer-kernels.yaml` (known blind spots).
    #[test]
    fn no_row_allocates_inside_run() {
        let _budget = hold_budget();
        backend::fork::set_threads(1);
        backend::upstream::set_threads(1);
        if vendor::NAME != "none" {
            vendor::set_threads(1).unwrap();
        }
        // The cells a primary ratio is formed from: column-major everywhere (the vendor's driver
        // route), plus row-major `gemm`, the only family the vendor also measures row-major. A
        // row-major LAPACK-driver cell has no vendor counterpart (it is a skip, `lapack.rs`), and
        // the faer row may allocate its own column-major copy inside the library call there
        // (`lu/partial_pivoting/factor.rs:396-418`), a declared blind spot in
        // benchmarks/suites/faer-kernels.yaml, not a harness allocation.
        for (order, families) in [(Order::F, FAMILIES), (Order::C, &["gemm"][..])] {
            for row in crate::engine_rows() {
                for family in families {
                    for dtype in [Dtype::F64, Dtype::C64] {
                        let case = Case {
                            m: 8,
                            n: 8,
                            k: if *family == "gemm" { Some(8) } else { None },
                            batch: 1,
                        };
                        let ops = match crate::ops_for(&row, dtype, family, case, order) {
                            Ok(ops) => ops,
                            // a route the arm does not have (a vendor C-order driver cell, a
                            // family with no LAPACK route) is a skip, never a measurement
                            Err(_) => continue,
                        };
                        for mut op in ops {
                            // The `api-inclusive` row is the owning route by definition: the
                            // library allocates its outputs and its scratch inside the timed call.
                            // It is a labelled secondary row and is excluded here; every primary
                            // (empty-note) row must be allocation-free.
                            if op.note == "api-inclusive" {
                                continue;
                            }
                            op.prepare().unwrap();
                            let _ = alloc_probe::take();
                            op.run(Token::none()).unwrap();
                            let (allocs, frees) = alloc_probe::take();
                            assert_eq!(
                                (allocs, frees),
                                (0, 0),
                                "{row} {family} {} {}: {allocs} allocations and {frees} frees \
                                 inside the timed run",
                                dtype.label(),
                                order.label(),
                            );
                            op.check().unwrap();
                        }
                    }
                }
            }
        }
    }

    /// **A LAPACK driver is called column-major only.** LAPACKE's row-major path allocates a
    /// transposed copy and transposes in and out of it inside the call, in every `_work` variant
    /// (reference `lapacke_dgeqrf_work.c:70-92`), so a C-order driver cell has no route that
    /// leaves the clock free of allocation and conversion. The vendor row must therefore report
    /// that cell as skipped rather than measure it, while `gemm` keeps both orders (CBLAS's
    /// row-major entry point moves no data).
    #[test]
    fn vendor_drivers_are_column_major_only() {
        let _budget = hold_budget();
        if vendor::NAME == "none" {
            return;
        }
        let row = crate::lapack::row_name().to_owned();
        for family in DRIVER_FAMILIES {
            let cell = |order| {
                crate::ops_for(
                    &row,
                    Dtype::F64,
                    family,
                    Case {
                        m: 8,
                        n: 8,
                        k: None,
                        batch: 1,
                    },
                    order,
                )
            };
            let error = match cell(Order::C) {
                Ok(_) => panic!("{family}: a C-order LAPACK driver cell must be a skip"),
                Err(error) => error,
            };
            assert!(
                error.contains("column-major") && error.contains("c order"),
                "{family}: the skip must name the reason, got {error:?}"
            );
            cell(Order::F).unwrap_or_else(|e| panic!("{family} in f order: {e}"));
        }
        for order in [Order::C, Order::F] {
            crate::ops_for(
                &row,
                Dtype::F64,
                "gemm",
                Case {
                    m: 8,
                    n: 8,
                    k: Some(8),
                    batch: 1,
                },
                order,
            )
            .unwrap_or_else(|e| panic!("gemm {}: {e}", order.label()));
        }
    }

    /// **The page divides like with like.** The report's cell key includes the `note`
    /// (`campaign_toolkit/timing_evidence.py`, `report_run.py`), so the note is what decides
    /// whether two rows are put in one ratio. Two rules follow and are checked here:
    ///
    /// * a faer row that times two routes emits the owning one as `api-inclusive` and the
    ///   allocation-free one with an EMPTY note, so the vendor row (empty note) can only pair with
    ///   the allocation-free route -- in-place with in-place, never owning with in-place;
    /// * every vendor route carries the empty note, so it never lands in the API-cost row.
    #[test]
    fn notes_pair_the_same_contract() {
        let _budget = hold_budget();
        let case = Case {
            m: 8,
            n: 8,
            k: None,
            batch: 1,
        };
        for row in ["faer-fork", "faer-upstream"] {
            for family in TWO_ROUTE_FAMILIES {
                let ops = crate::ops_for(row, Dtype::F64, family, case, Order::F).unwrap();
                assert_eq!(ops.len(), 2, "{row} {family} must time both routes");
                assert_eq!(
                    ops[0].note, "api-inclusive",
                    "{row} {family}: the owning route is the labelled API-cost row"
                );
                assert_eq!(
                    ops[1].note, "",
                    "{row} {family}: the allocation-free route is the primary row"
                );
            }
        }
        if vendor::NAME == "none" {
            return;
        }
        let row = crate::lapack::row_name().to_owned();
        for family in DRIVER_FAMILIES.iter().copied().chain(["gemm"]) {
            let case = Case {
                m: 8,
                n: 8,
                k: if family == "gemm" { Some(8) } else { None },
                batch: 1,
            };
            let ops = crate::ops_for(&row, Dtype::F64, family, case, Order::F).unwrap();
            assert_eq!(ops.len(), 1, "{family}");
            assert_eq!(
                ops[0].note, "",
                "{family}: a vendor route is always the primary, allocation-free row"
            );
        }
    }
}
