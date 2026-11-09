//! Time-based priming followed by every timed repetition, in execution order.
//!
//! `prepare` is outside each clocked region. All providers use this loop;
//! equivalent work inside their `run` closures must be audited separately.
//!
//! This module also owns the cell clock: the per-cell time limit (#57) that stops one
//! cell that cannot finish from consuming a campaign.

use std::cell::Cell;
use std::time::{Duration, Instant};

use crate::op::{Op, Token};

thread_local! {
    /// The deadline of the cell running on this thread, if one is armed. A cell is executed on one
    /// thread from its first `prepare` to its check, so the clock belongs to the thread; the
    /// reference math the check calls runs on that same thread.
    static DEADLINE: Cell<Option<Instant>> = const { Cell::new(None) };
    /// Set when this thread's cell passed its limit, so the executor can tell a cut cell from one
    /// that failed on its own.
    static CUT: Cell<bool> = const { Cell::new(false) };
}

/// Start the clock of one cell. `None` leaves it disarmed.
///
/// The executor fixes an absolute deadline before the cell's construction and arms with
/// [`arm_at`]; this relative form is what the reference-math tests use.
#[cfg(test)]
pub fn arm(limit: Option<Duration>) {
    arm_at(deadline_in(limit));
}

/// The absolute deadline `start + limit`, for a cell whose clock is armed on more than one thread.
pub fn deadline_at(start: Instant, limit: Option<Duration>) -> Option<Instant> {
    let limit = limit?;
    if let Some(deadline) = start.checked_add(limit) {
        return Some(deadline);
    }
    // A limit the platform clock cannot reach is still a limit, and `None` means "no
    // limit" to every caller: saturate at one year, far past any cell the harness waits for.
    const SATURATED: Duration = Duration::from_secs(365 * 24 * 60 * 60);
    Some(
        start
            .checked_add(SATURATED)
            .expect("a one-year deadline is representable on this clock"),
    )
}

/// The absolute deadline `limit` from now.
#[cfg(test)]
pub fn deadline_in(limit: Option<Duration>) -> Option<Instant> {
    deadline_at(Instant::now(), limit)
}

/// Arm the clock with an already computed deadline. A cell may be armed on the caller before its
/// reference construction and then on the worker that runs it, and both must read one deadline.
pub fn arm_at(deadline: Option<Instant>) {
    DEADLINE.with(|slot| slot.set(deadline));
    CUT.with(|cut| cut.set(false));
}

pub fn disarm() {
    DEADLINE.with(|deadline| deadline.set(None));
}

/// Has the armed limit passed? Polled by the loops the harness itself owns -- the timing loop and
/// the serial reference math a check runs (`shared::gram_residual` and friends) -- so a cell that
/// cannot finish returns to the executor instead of running for a weekend. A single library call
/// that is already in progress cannot be interrupted: the cell is cut at the next poll point, and
/// every loop that can grow without bound has one.
pub fn expired() -> bool {
    DEADLINE.with(|deadline| match deadline.get() {
        None => false,
        Some(deadline) if Instant::now() < deadline => false,
        Some(_) => {
            CUT.with(|cut| cut.set(true));
            true
        }
    })
}

/// The error a harness-owned loop returns when it stops at its cell's limit. The `CUT` flag says
/// so, so the executor records a `timeout` row rather than a failure of the library call.
pub fn cut_error() -> String {
    "the cell limit was reached before the next call".to_owned()
}

/// Whether the cell that just returned was cut off by its limit, rather than failing on its own.
pub fn cut() -> bool {
    CUT.with(|cut| cut.get())
}

/// A stage's result, with the limit checked before the stage's own error is reported: a call that
/// returned an error after the limit had passed is a cut cell, not a library failure, and the
/// `expired` poll here is what records the cut.
fn stage<T>(result: Result<T, String>) -> Result<T, String> {
    match result {
        Err(e) => {
            if expired() {
                return Err(format!("{e}; the cell limit was reached"));
            }
            Err(e)
        }
        ok => ok,
    }
}

/// What one cell measured.
#[derive(Debug)]
pub struct Measured {
    /// One wall-clock millisecond count per timed call, in execution order.
    pub ms: Vec<f64>,
    /// The check that runs once after them, outside the clock. Reported apart from `ms` because a
    /// check that costs more than the measurement it guards is what made one `n = 4096` cell read
    /// as a 27 h benchmark: at `n = 512` four cells recorded 108, 98, 99 and 98 ms of timed region
    /// while the process took 5.8 s.
    pub check_ms: f64,
}

/// Prime for `prime_ms`, then return all `reps` times in milliseconds, and the time the op's
/// `check` then costs outside the clock.
///
/// `token` is the row's parallelism, lent to every `run`: the measured library's `Par` is exclusive
/// authority from 0.25 on, so the loop owns it and reborrows per call.
pub fn measure(
    op: &mut Op,
    prime_ms: u64,
    reps: usize,
    token: &mut Token<'_>,
) -> Result<Measured, String> {
    if reps == 0 {
        return Err("--reps must be positive".to_owned());
    }
    let mut ms = Vec::with_capacity(reps);
    let until = Instant::now() + Duration::from_millis(prime_ms);
    while Instant::now() < until {
        if expired() {
            return Err("the cell limit was reached while priming".to_owned());
        }
        stage(op.prepare())?;
        if expired() {
            return Err("the cell limit was reached while priming".to_owned());
        }
        stage(op.run(token.reborrow()))?;
    }
    for _ in 0..reps {
        if expired() {
            return Err("the cell limit was reached before every timed call ran".to_owned());
        }
        stage(op.prepare())?;
        // `prepare` is outside the timed region but inside the cell, and it is not free: the
        // `householder_apply` routes factorize here. A call that starts after it crossed the
        // limit is a call the limit did not stop.
        if expired() {
            return Err("the cell limit was reached while preparing".to_owned());
        }
        let start = Instant::now();
        stage(op.run(token.reborrow()))?;
        let elapsed = start.elapsed();
        ms.push(elapsed.as_secs_f64() * 1e3);
    }
    // Poll once more after the last repetition: a call that ran past the limit returns `Ok`, and
    // "it finished" is not the same as "it finished inside its budget". Without this, the last
    // repetition could exceed the limit and the cell would still be accepted as a measurement.
    if expired() {
        return Err("the cell limit was reached during the last timed call".to_owned());
    }
    let start = Instant::now();
    stage(op.check())?;
    let check_ms = start.elapsed().as_secs_f64() * 1e3;
    // Same for the check: a successful check that ran past the limit must not become an `ok` row.
    if expired() {
        return Err("the cell limit was reached while the check ran".to_owned());
    }
    stage(op.prepare())?;
    // The restore after the check is a prepare too: a cell that crossed its limit there did not
    // finish inside its budget, and must not be written as an `ok` row.
    if expired() {
        return Err("the cell limit was reached while preparing".to_owned());
    }
    Ok(Measured { ms, check_ms })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::op_from;
    use std::sync::{Arc, Mutex};

    /// A last repetition or a check that returns `Ok` after the limit is not a measurement: both
    /// post-stage polls must reject it. Removing either poll accepts the cell as `ok`.
    #[test]
    fn a_successful_stage_past_the_limit_is_not_a_measurement() {
        // A run that outlives the limit: the post-repetition poll rejects it.
        let mut op = op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| {
                std::thread::sleep(Duration::from_millis(30));
                Ok(())
            },
            |_| Ok(()),
        );
        arm(Some(Duration::from_millis(5)));
        let out = measure(&mut op, 0, 1, &mut Token::none());
        assert!(
            out.is_err(),
            "a run past the limit must not be accepted as a measurement"
        );
        assert!(cut());
        disarm();

        // A check that outlives the limit: the post-check poll rejects it.
        let mut op = op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| Ok(()),
            |_| {
                std::thread::sleep(Duration::from_millis(30));
                Ok(())
            },
        );
        arm(Some(Duration::from_millis(5)));
        let out = measure(&mut op, 0, 1, &mut Token::none());
        assert!(
            out.is_err(),
            "a check past the limit must not turn the row into `ok`"
        );
        assert!(cut());
        disarm();
    }

    /// `prepare` is outside the timed region but inside the cell, and it can cross the limit
    /// (`householder_apply` factorizes there). The call after it must not run, and the restore that
    /// follows the check is a `prepare` too: if it crosses, the cell is cut rather than accepted.
    #[test]
    fn a_timed_prepare_past_the_limit_does_not_run() {
        // A timed prepare that crosses the limit: no run may follow it.
        let runs = Arc::new(Mutex::new(0usize));
        let r = runs.clone();
        let mut op = op_from(
            (),
            String::new(),
            |_| {
                std::thread::sleep(Duration::from_millis(30));
                Ok(())
            },
            move |_, _tk| {
                *r.lock().unwrap() += 1;
                Ok(())
            },
            |_| Ok(()),
        );
        arm(Some(Duration::from_millis(5)));
        assert!(
            measure(&mut op, 0, 1, &mut Token::none()).is_err(),
            "a prepare past the limit must stop the cell"
        );
        assert_eq!(
            *runs.lock().unwrap(),
            0,
            "the timed call must not start after prepare crossed the limit"
        );
        assert!(cut());
        disarm();
    }

    #[test]
    fn the_final_prepare_past_the_limit_cuts_the_cell() {
        // The final prepare (restore after the check) crosses the limit: the cell is cut.
        let calls = Arc::new(Mutex::new(0usize));
        let c = calls.clone();
        let mut op = op_from(
            (),
            String::new(),
            move |_| {
                let mut n = c.lock().unwrap();
                *n += 1;
                if *n == 2 {
                    std::thread::sleep(Duration::from_millis(30));
                }
                Ok(())
            },
            |_, _tk| Ok(()),
            |_| Ok(()),
        );
        arm(Some(Duration::from_millis(5)));
        assert!(
            measure(&mut op, 0, 1, &mut Token::none()).is_err(),
            "a final prepare past the limit must cut the cell, not finish it"
        );
        assert!(cut());
        disarm();
    }

    /// A stage that fails after the limit is a cut, not a library failure: `stage` polls before it
    /// reports the stage's own error. Without that poll this cell is `failed`.
    #[test]
    fn a_stage_error_after_the_limit_is_a_cut() {
        let mut op = op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| {
                std::thread::sleep(Duration::from_millis(30));
                Err("the library said no".to_owned())
            },
            |_| Ok(()),
        );
        arm(Some(Duration::from_millis(5)));
        assert!(measure(&mut op, 0, 1, &mut Token::none()).is_err());
        assert!(cut(), "an error after the limit must be recorded as a cut");
        disarm();
    }

    #[test]
    fn retains_repetitions_and_checks_the_last_run() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let mut op = op_from(
            events.clone(),
            String::new(),
            |e| {
                e.lock().unwrap().push("prepare");
                Ok(())
            },
            |e, _tk| {
                e.lock().unwrap().push("run");
                Ok(())
            },
            |e| {
                e.lock().unwrap().push("check");
                Ok(())
            },
        );
        assert!(measure(&mut op, 0, 0, &mut Token::none()).is_err());
        assert!(events.lock().unwrap().is_empty());
        let measured = measure(&mut op, 0, 3, &mut Token::none()).unwrap();
        assert_eq!(measured.ms.len(), 3);
        assert!(measured.ms.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(measured.check_ms >= 0.0);
        assert_eq!(
            *events.lock().unwrap(),
            ["prepare", "run", "prepare", "run", "prepare", "run", "check", "prepare"]
        );
        let mut failed = op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| Ok(()),
            |_| Err("bad result".to_owned()),
        );
        assert_eq!(
            measure(&mut failed, 0, 3, &mut Token::none()).unwrap_err(),
            "bad result"
        );
    }

    /// A cell that never finishes is cut, and the loop reports it as an error instead of waiting
    /// for it. Without the cell clock this test does not return.
    #[test]
    fn the_cell_clock_cuts_a_check_that_outruns_it() {
        arm(Some(Duration::from_millis(50)));
        let mut slow = op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| Ok(()),
            |_| {
                while !expired() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err("residual infinite".to_owned())
            },
        );
        let started = Instant::now();
        assert!(measure(&mut slow, 0, 1, &mut Token::none()).is_err());
        assert!(cut(), "the cell clock did not record the cut");
        assert!(started.elapsed() < Duration::from_secs(30));
        disarm();
        assert!(!expired());
        // and an unarmed clock never cuts anything
        arm(None);
        assert!(!expired());
        assert!(!cut());
    }

    /// The check's time is reported apart from the timed calls, which is what made the 27 h stall
    /// visible in the record.
    #[test]
    fn the_check_time_is_reported_apart_from_the_timed_calls() {
        arm(None);
        let mut op = op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| Ok(()),
            |_| {
                std::thread::sleep(Duration::from_millis(30));
                Ok(())
            },
        );
        let measured = measure(&mut op, 0, 2, &mut Token::none()).unwrap();
        assert_eq!(measured.ms.len(), 2);
        assert!(
            measured.check_ms >= 25.0,
            "check took {} ms, which is the sleep it did",
            measured.check_ms
        );
        assert!(
            measured.ms.iter().all(|ms| *ms < 25.0),
            "the check's sleep leaked into the timed region: {:?}",
            measured.ms
        );
    }
}
