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
    Fork(faer_fork::Par<'a>),
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
        use faer_fork::utils::team::stats;
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
                let n = if family == "triangular_solve" { 2048 } else { 256 };
                let case = Case {
                    m: n,
                    n,
                    k: if family == "gemm" { Some(n) } else { None },
                    batch: 1,
                };
                for mut op in crate::ops_for("faer-fork", Dtype::F64, family, case, order).unwrap() {
                    op.prepare().unwrap();
                    let s = faer_fork::team::with_team(threads, |par| {
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
                    assert!(team > 0, "faer-fork {family} {} {}: no team dispatch ({s:?})", order.label(), op.note);
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
                if !row.starts_with("faer-")
                    && matches!(
                        family,
                        "full_piv_lu" | "triangular_solve" | "householder_apply"
                    )
                {
                    continue;
                }
                for dtype in [Dtype::F64, Dtype::C64] {
                    for order in [Order::C, Order::F] {
                        let case = Case {
                            m: 8,
                            n: 8,
                            k: if family == "gemm" { Some(8) } else { None },
                            batch: 2,
                        };
                        let ops = crate::ops_for(&row, dtype, family, case, order).unwrap();
                        // householder_apply times both public forms; both faer rows time `lu`,
                        // `cholesky` and `solve` through the owning method and the `in_place` route.
                        let expected = match family {
                            "householder_apply" => 2,
                            "lu" | "cholesky" | "solve" if row.starts_with("faer-") => 2,
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
}
