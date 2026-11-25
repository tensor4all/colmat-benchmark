//! Time-based priming followed by every timed repetition, in execution order, then the check of
//! the last timed repetition's result.
//!
//! `prepare` is outside each clocked region. All providers use this loop;
//! equivalent work inside their `run` closures must be audited separately.

use std::time::{Duration, Instant};

use crate::op::{Op, Token};

/// Where one cell's untimed wall time went, for `FBENCH_PHASES=1` diagnostics. Every field is read
/// from `Instant`s taken *between* clocked regions -- before the first and after the last timed
/// repetition, and around the check -- so recording it adds no instruction to any clocked region.
#[derive(Clone, Copy, Debug, Default)]
pub struct Phases {
    /// wall time of the priming loop (`prepare` + `run`, repeated for `prime_ms`)
    pub prime: Duration,
    /// wall time of the whole timed loop, its per-repetition `prepare` included
    pub timed_loop: Duration,
    /// the sum of the recorded samples (the clocked part of `timed_loop`)
    pub sampled: Duration,
    /// the `check` of the last timed repetition's result
    pub check: Duration,
    /// the final `prepare` that leaves the inputs restored
    pub restore: Duration,
}

/// Prime for `prime_ms`, then return all `reps` times in milliseconds, and check the result the
/// **last timed repetition** left behind: the measured call itself, on the row's own token, not a
/// separate execution. The harness calls the two halves ([`time_loop`], [`check_last`]) itself so it
/// can release a row's reservation in between (`main.rs` `measure_row`); this is their composition,
/// in the order the harness runs them.
///
/// `token` is the row's parallelism, lent to every `run`: the measured library's `Par` is exclusive
/// authority from 0.25 on, so the loop owns it and reborrows per call.
#[cfg(test)]
pub fn measure(
    op: &mut Op,
    prime_ms: u64,
    reps: usize,
    token: &mut Token<'_>,
) -> Result<Vec<f64>, String> {
    let mut phases = Phases::default();
    let samples = time_loop(op, prime_ms, reps, token, &mut phases)?;
    check_last(op, &mut phases)?;
    Ok(samples)
}

/// The priming loop and the timed repetitions, nothing after them: the op's state is the one the
/// last timed `run` left, and [`check_last`] must be called on it before the next `prepare`.
///
/// Split from the check so that a row holding a reservation (a `colmat` team, a spindle worker
/// group) can release it before the check: the check's reference products run on the Rayon pool
/// (`shared::mm`), the same cores a held team would otherwise be occupying.
pub fn time_loop(
    op: &mut Op,
    prime_ms: u64,
    reps: usize,
    token: &mut Token<'_>,
    phases: &mut Phases,
) -> Result<Vec<f64>, String> {
    if reps == 0 {
        return Err("--reps must be positive".to_owned());
    }
    let mut samples = Vec::with_capacity(reps);
    let primed = Instant::now();
    let until = primed + Duration::from_millis(prime_ms);
    while Instant::now() < until {
        op.prepare()?;
        op.run(token.reborrow())?;
    }
    let looped = Instant::now();
    phases.prime = looped - primed;
    for _ in 0..reps {
        op.prepare()?;
        let start = Instant::now();
        op.run(token.reborrow())?;
        let elapsed = start.elapsed();
        samples.push(elapsed.as_secs_f64() * 1e3);
    }
    phases.timed_loop = looped.elapsed();
    phases.sampled = Duration::from_secs_f64(samples.iter().sum::<f64>() / 1e3);
    Ok(samples)
}

/// Check the result the last timed repetition left behind -- the measured call itself, at the
/// row's own parallelism, not a separate execution -- then restore the inputs.
pub fn check_last(op: &mut Op, phases: &mut Phases) -> Result<(), String> {
    let checked = Instant::now();
    op.check()?;
    let restored = Instant::now();
    phases.check = restored - checked;
    op.prepare()?;
    phases.restore = restored.elapsed();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::op_from;
    use std::sync::{Arc, Mutex};

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
        let samples = measure(&mut op, 0, 3, &mut Token::none()).unwrap();
        assert_eq!(samples.len(), 3);
        assert!(samples.iter().all(|v| v.is_finite() && *v >= 0.0));
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
}
