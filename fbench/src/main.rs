//! `fbench`: the measurement harness for the faer benchmark campaign.
//!
//! Three subcommands, at the contract `scripts/record_run.py` reads:
//!
//! * `info` prints one `key=value` per line, naming the vendor library the binary
//!   actually linked and how;
//! * `verify` checks every requested case, on every arm, and ends with exactly
//!   `all comparisons within tolerance` when every case passed;
//! * `run` measures and writes the CSV the report reads.
//!
//! One binary links one vendor library, so the LAPACK row's name (`lapack-mkl`
//! or `lapack-openblas`) is derived from what was linked, and a family a provider
//! cannot take is written as `skipped` rather than omitted.

// The numerical reference math and the adapters index matrices by `(row, column)`
// through an explicit linear index because the layout is a measured axis; iterators
// would obscure the stride arithmetic that is the point of those loops.
#![allow(clippy::needless_range_loop)]

mod backend;
mod lapack;
mod op;
mod shared;
mod timing;
mod vendor;

// The vendor source crates only carry native link directives; refer to them so the
// linker keeps the `-l` flags their build scripts emit instead of pruning an unused
// dependency. Without this a build succeeds but the CBLAS/LAPACKE symbols are absent.
#[cfg(any(feature = "link-mkl", feature = "link-openblas"))]
use blas_src as _;
#[cfg(any(feature = "link-mkl", feature = "link-openblas"))]
use lapack_src as _;

use shared::{Case, Dtype, Order};
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

const USAGE: &str = "usage: fbench info [--threads N] | verify --threads N --n LIST [--batch LIST] [--dtype LIST] [--family LIST] [--regime R] [--rows LIST] [--cell-timeout SECS] [--csv PATH] | run --threads N --n LIST [--batch LIST] [--dtype LIST] [--family LIST] [--regime R] [--rows LIST] [--reps M] [--prime-ms MS] [--rotate-rows K] [--statistic best|median] [--cell-timeout SECS] --csv PATH";

/// The per-cell limit when neither `--cell-timeout` nor the environment names one.
///
/// A cell is one `Op` and one CSV row. The value is a starting point, not a constant written into
/// the loop: the campaign gets it without composing an extra argument, and a host or a pilot that
/// knows its cells are cheaper lowers it with `--cell-timeout` or `FBENCH_CELL_TIMEOUT`.
const DEFAULT_CELL_TIMEOUT_S: u64 = 3600;
const CELL_TIMEOUT_ENV: &str = "FBENCH_CELL_TIMEOUT";

/// Every family name the suite declares, in the harness's priority order.
const ALL_FAMILIES: &[&str] = &[
    "gemm",
    "qr_factor",
    "qr",
    "lu",
    "cholesky",
    "solve",
    "triangular_solve",
    "svd_values",
    "svd_thin",
    "eigh",
    "full_piv_lu",
    "upgrade_factor",
    "householder_apply",
    // Declared by the suite and not implemented yet: every `run` emits a `skipped` row
    // for it rather than omitting it, which is the contract the recorder reads. It has
    // to be *known* here for that to happen -- an unrecognised name is a usage error.
    "eig",
];

#[derive(Clone, Copy)]
struct RawShape {
    m: usize,
    n: usize,
    k: Option<usize>,
}

impl RawShape {
    fn parse(text: &str) -> Result<Self, String> {
        let parts: Vec<&str> = text.split('x').collect();
        let num = |s: &str| {
            s.parse::<usize>()
                .map_err(|_| format!("invalid shape {text:?}"))
        };
        match parts.as_slice() {
            [n] => {
                let n = num(n)?;
                Ok(RawShape { m: n, n, k: None })
            }
            [m, n] => Ok(RawShape {
                m: num(m)?,
                n: num(n)?,
                k: None,
            }),
            [m, k, n] => Ok(RawShape {
                m: num(m)?,
                n: num(n)?,
                k: Some(num(k)?),
            }),
            _ => Err(format!("invalid shape {text:?}")),
        }
    }

    fn case(self, family: &str, batch: usize) -> Result<Case, String> {
        if family == "gemm" {
            if self.k.is_none() && self.m != self.n {
                return Err(format!(
                    "gemm needs MxKxN; {} is a rectangular MxN shape",
                    self.shape_text()
                ));
            }
            Ok(Case {
                m: self.m,
                n: self.n,
                k: Some(self.k.unwrap_or(self.m)),
                batch,
            })
        } else if self.k.is_some() {
            Err(format!(
                "{} is a three-index shape, which only gemm declares; {} takes MxN",
                self.shape_text(),
                family
            ))
        } else {
            Ok(Case {
                m: self.m,
                n: self.n,
                k: None,
                batch,
            })
        }
    }

    fn shape_text(self) -> String {
        match self.k {
            Some(k) => format!("{}x{}x{}", self.m, k, self.n),
            None if self.m == self.n => self.n.to_string(),
            None => format!("{}x{}", self.m, self.n),
        }
    }
}

#[derive(Clone, Copy)]
enum Statistic {
    Best,
    Median,
}

/// The timed loop of one row.
///
/// The two faer rows run it with the row's parallelism held around the whole loop, once, so that
/// priming, every repetition and the check reuse one reservation and its setup is outside every
/// clocked region. The vendor rows own their threading and are not touched.
///
/// * `faer-fork` with the team API (`fork-team`): `faer::team::with_team(threads, ..)` owns a
///   `threadteam` team for the loop and its `Par::Team` is lent to each call - that is how a caller
///   with a thread budget calls this library now (the crate docs, *parallelism*), and the token can
///   only be borrowed, never copied, from 0.25 on.
/// * `faer-fork` without it (0.24/0.25 with the Rayon route): the spindle worker group, as before.
/// * `faer-upstream`: the spindle worker group, which is how the unchanged row has always been
///   measured.
fn measure_row(
    row: &str,
    threads: usize,
    ops: &mut op::Op,
    prime_ms: u64,
    reps: usize,
    deadline: Option<Instant>,
) -> (Result<timing::Measured, String>, bool) {
    #[cfg(feature = "fork-team")]
    if threads > 1 && row == "faer-fork" {
        return faer_fork::team::with_team(threads, |par| {
            let mut token = op::Token::Fork(par);
            measure_armed(ops, prime_ms, reps, &mut token, deadline)
        });
    }
    if threads > 1 && (row == "faer-fork" || row == "faer-upstream") {
        spindle::with_lock(threads, || {
            measure_armed(ops, prime_ms, reps, &mut op::Token::none(), deadline)
        })
    } else {
        measure_armed(ops, prime_ms, reps, &mut op::Token::none(), deadline)
    }
}

/// Arm the cell clock on the thread that actually executes the cell, run it there, and report
/// whether the limit cut it.
///
/// The clock is thread-local (`timing::DEADLINE`), and `with_team`/`with_lock` run the closure on a
/// team member or a worker, not on the caller: a clock armed on the caller would never be read by
/// the thread doing the work, and every multithreaded row would run unlimited. The `CUT` flag has
/// the same owner, so it is read here, on that thread, and returned rather than re-read later.
fn measure_armed(
    ops: &mut op::Op,
    prime_ms: u64,
    reps: usize,
    token: &mut op::Token<'_>,
    deadline: Option<Instant>,
) -> (Result<timing::Measured, String>, bool) {
    timing::arm_at(deadline);
    let measured = timing::measure(ops, prime_ms, reps, token);
    let cut = timing::cut();
    timing::disarm();
    (measured, cut)
}

/// An `ok` row is a measurement only if the numbers it *derives* are finite. Every field is checked
/// where it is parsed (`parse_number`), but `per_item_us` scales the summary by 1000 and the median
/// averages two samples, so finite inputs can still overflow: `1e308` samples became `inf` in the
/// CSV on an `ok` row. Used by `finish`, which writes every row a cell protocol produces.
fn derived_ok_values_are_finite(
    cell: &Cell,
    batch: usize,
    statistic: Statistic,
) -> Result<(), String> {
    let total_ms = summarize(&cell.samples, statistic);
    let per_item_us = total_ms.map(|ms| ms * 1000.0 / batch as f64);
    for (name, value) in [("total_ms", total_ms), ("per_item_us", per_item_us)] {
        if let Some(value) = value {
            if !value.is_finite() {
                return Err(format!(
                    "the {name} this cell derives is not finite ({value}); the raw fields are"
                ));
            }
        }
    }
    Ok(())
}

fn summarize(samples: &[f64], statistic: Statistic) -> Option<f64> {
    if samples.is_empty() {
        return None;
    }
    match statistic {
        Statistic::Best => samples.iter().copied().reduce(f64::min),
        Statistic::Median => {
            let mut ordered = samples.to_vec();
            ordered.sort_by(f64::total_cmp);
            let middle = ordered.len() / 2;
            Some(if ordered.len() % 2 == 0 {
                (ordered[middle - 1] + ordered[middle]) * 0.5
            } else {
                ordered[middle]
            })
        }
    }
}

struct Args {
    command: String,
    threads: Option<usize>,
    shapes: Vec<RawShape>,
    batches: Vec<usize>,
    dtypes: Vec<Dtype>,
    families: Vec<String>,
    regime: String,
    reps: usize,
    prime_ms: u64,
    /// Explicit row-order probe, not a guarantee of equivalent conditions.
    rotate_rows: usize,
    /// Which engine rows this run measures. A campaign may declare a subset per arm
    /// (an arm that carries an unchanged faer row twice does not have to pay for both).
    rows: Vec<String>,
    statistic: Statistic,
    /// The per-cell limit in seconds; `0` leaves cells uncut. Resolved from `--cell-timeout`,
    /// then `FBENCH_CELL_TIMEOUT`, then [`DEFAULT_CELL_TIMEOUT_S`].
    cell_timeout: u64,
    csv: Option<String>,
    /// The hardware profile the recorder declared (`FBENCH_PROFILE`), written into every
    /// row: a CSV that has to be found in the campaign directory to learn which machine
    /// produced it is a number without its machine. Empty when `fbench` is run by hand.
    profile: String,
}

fn parse(args: &[String]) -> Result<Args, String> {
    if args.len() < 2 {
        return Err(USAGE.to_owned());
    }
    let command = args[1].clone();
    if command == "info" {
        let mut threads = None;
        let mut i = 2;
        while i < args.len() {
            if args[i] == "--threads" {
                i += 1;
                threads = Some(
                    args.get(i)
                        .ok_or(USAGE)?
                        .parse()
                        .map_err(|_| USAGE.to_owned())?,
                );
            } else {
                return Err(USAGE.to_owned());
            }
            i += 1;
        }
        return Ok(Args {
            command,
            threads,
            shapes: vec![],
            batches: vec![],
            dtypes: vec![],
            families: vec![],
            regime: "unspecified".to_owned(),
            reps: 5,
            prime_ms: 500,
            rotate_rows: 0,
            rows: Vec::new(),
            statistic: Statistic::Best,
            cell_timeout: 0,
            csv: None,
            profile: String::new(),
        });
    }
    if command != "run" && command != "verify" {
        return Err(USAGE.to_owned());
    }
    let mut a = Args {
        command,
        threads: None,
        shapes: vec![],
        batches: vec![1],
        dtypes: vec![Dtype::F64],
        families: ALL_FAMILIES.iter().map(|f| (*f).to_owned()).collect(),
        regime: "unspecified".to_owned(),
        reps: 5,
        prime_ms: 500,
        rotate_rows: 0,
        rows: Vec::new(),
        statistic: Statistic::Best,
        cell_timeout: 0,
        csv: None,
        profile: String::new(),
    };
    let mut cell_timeout: Option<u64> = None;
    let mut i = 2;
    while i < args.len() {
        let flag = args[i].clone();
        let mut value = || -> Result<String, String> {
            i += 1;
            args.get(i).cloned().ok_or_else(|| USAGE.to_owned())
        };
        match flag.as_str() {
            "--threads" => {
                a.threads = Some(value()?.parse().map_err(|_| USAGE.to_owned())?);
            }
            "--n" => {
                a.shapes = value()?
                    .split(',')
                    .map(RawShape::parse)
                    .collect::<Result<_, _>>()?;
            }
            "--batch" => {
                a.batches = value()?
                    .split(',')
                    .map(|v| v.parse::<usize>().map_err(|_| USAGE.to_owned()))
                    .collect::<Result<_, _>>()?;
            }
            "--dtype" => {
                a.dtypes = value()?
                    .split(',')
                    .map(|d| match d {
                        "f64" => Ok(Dtype::F64),
                        "c64" => Ok(Dtype::C64),
                        "f32" => Ok(Dtype::F32),
                        "c32" => Ok(Dtype::C32),
                        other => Err(format!("unknown dtype {other:?}")),
                    })
                    .collect::<Result<_, _>>()?;
            }
            "--family" => {
                a.families = value()?.split(',').map(str::to_owned).collect();
                for family in &a.families {
                    if !ALL_FAMILIES.contains(&family.as_str()) {
                        return Err(format!("unknown family {family:?}"));
                    }
                }
            }
            "--regime" => a.regime = value()?,
            "--rows" => {
                a.rows = value()?
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(str::to_owned)
                    .collect();
                let known = engine_rows();
                for row in &a.rows {
                    if !known.contains(row) {
                        return Err(format!(
                            "unknown row {row:?}; this binary carries {known:?}"
                        ));
                    }
                }
            }
            "--rotate-rows" => {
                a.rotate_rows = value()?.parse().map_err(|e| format!("{e}"))?;
            }
            "--statistic" => {
                a.statistic = match value()?.as_str() {
                    "best" => Statistic::Best,
                    "median" => Statistic::Median,
                    _ => return Err("--statistic must be best or median".to_owned()),
                };
            }
            "--reps" => a.reps = value()?.parse().map_err(|_| USAGE.to_owned())?,
            "--prime-ms" => a.prime_ms = value()?.parse().map_err(|_| USAGE.to_owned())?,
            "--cell-timeout" => {
                cell_timeout = Some(
                    value()?
                        .parse()
                        .map_err(|_| "--cell-timeout is not a number of seconds".to_owned())?,
                );
            }
            "--csv" => a.csv = Some(value()?),
            _ => return Err(USAGE.to_owned()),
        }
        i += 1;
    }
    if a.shapes.is_empty() {
        // No `--n` means the default benchmark: the 64 cell of every family. `RawShape::case`
        // turns `64` into `64x64x64` for `gemm`, so one raw shape covers both shapes.
        a.shapes = vec![RawShape::parse("64").expect("64 is a shape")];
    }
    if a.threads.is_none() {
        return Err(USAGE.to_owned());
    }
    if a.threads == Some(0) {
        return Err("--threads must be positive".to_owned());
    }
    if a.reps == 0 {
        return Err("--reps must be positive".to_owned());
    }
    if a.command == "run" && a.csv.is_none() {
        return Err("run needs --csv PATH".to_owned());
    }
    a.cell_timeout = match cell_timeout {
        Some(seconds) => seconds,
        None => match std::env::var(CELL_TIMEOUT_ENV) {
            Ok(text) => text
                .trim()
                .parse()
                .map_err(|_| format!("{CELL_TIMEOUT_ENV}={text:?} is not a number of seconds"))?,
            Err(std::env::VarError::NotPresent) => DEFAULT_CELL_TIMEOUT_S,
            Err(e) => return Err(format!("{CELL_TIMEOUT_ENV}: {e}")),
        },
    };
    if a.cell_timeout > 0 {
        let limit = Duration::from_secs(a.cell_timeout);
        if Instant::now().checked_add(limit).is_none() {
            return Err(format!(
                "--cell-timeout {} s is not representable on this clock: it would disarm the cell instead of limiting it",
                a.cell_timeout
            ));
        }
    }
    Ok(a)
}

/// The engine rows this binary carries: the two faer crates plus this build's
/// vendor. The vendor row is absent without a `link-*` feature, never guessed.
fn engine_rows() -> Vec<String> {
    let mut rows = vec!["faer-fork".to_owned(), "faer-upstream".to_owned()];
    if vendor::NAME != "none" {
        rows.push(lapack::row_name().to_owned());
    }
    rows
}

/// Build one cell's ops. The caller arms the cell's clock before this and keeps the same absolute
/// deadline for the run: construction is not free (`cholesky`/`solve` build their input with the
/// cubic `shared::mul_adj_b`, LAPACK `gemm` builds its whole reference product), it is part of the
/// cell, and a deadline recomputed at the start of the timed region would give that time back.
#[cfg(test)]
fn build_ops(
    row: &str,
    dtype: Dtype,
    family: &str,
    case: Case,
    order: Order,
) -> Result<Vec<op::Op>, String> {
    ops_for(row, dtype, family, case, order)
}

fn ops_for(
    row: &str,
    dtype: Dtype,
    family: &str,
    case: Case,
    order: Order,
) -> Result<Vec<op::Op>, String> {
    match row {
        "faer-fork" => backend::fork::ops(dtype, family, case, order),
        "faer-upstream" => backend::upstream::ops(dtype, family, case, order),
        _ => lapack::ops(dtype, family, case, order),
    }
}

fn info(threads: Option<usize>) {
    for (key, value) in vendor::identity().lines() {
        println!("{key}={value}");
    }
    println!("harness.name=fbench");
    println!(
        "host.logical_cpus={}",
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );
    if let Some(n) = threads {
        println!("threads.requested={n}");
        if n == 1 {
            println!("threads.faer=sequential");
        }
    }
    if threads.is_some() {
        println!("threads.rayon_observed={}", rayon::current_num_threads());
    }
    if let Some(n) = vendor::reported_threads() {
        println!("threads.vendor_reported={n}");
    }
}

fn configure_threads(n: usize) -> Result<(), String> {
    if n == 0 {
        return Err("--threads must be positive".to_owned());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| format!("cannot create the owned {n}-worker Rayon pool: {e}"))?;
    vendor::set_threads(n)?;
    backend::fork::set_threads(n);
    backend::upstream::set_threads(n);
    require_threads(n)
}

fn require_threads(n: usize) -> Result<(), String> {
    let observed = rayon::current_num_threads();
    if observed != n {
        return Err(format!("Rayon reports {observed} workers, expected {n}"));
    }
    vendor::require_threads(n)
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

struct Record {
    regime: String,
    family: String,
    dtype: &'static str,
    order: &'static str,
    m: usize,
    n: usize,
    k: Option<usize>,
    batch: usize,
    row: String,
    threads: usize,
    rayon_threads: usize,
    vendor_threads: Option<usize>,
    total_ms: Option<f64>,
    samples_ms: Vec<f64>,
    status: &'static str,
    note: String,
    profile: String,
    /// The measured region: every timed call of this cell, summed. Empty when the cell did not run.
    timed_ms: Option<f64>,
    /// The check that runs once after it, outside the clock. Empty when the cell did not run, or
    /// when it was cut before reaching the check.
    check_ms: Option<f64>,
}

impl Record {
    fn line(&self) -> String {
        let per_item = self
            .total_ms
            .map(|ms| format!("{:.6}", ms * 1000.0 / self.batch as f64))
            .unwrap_or_default();
        let total = self
            .total_ms
            .map(|ms| format!("{ms:.6}"))
            .unwrap_or_default();
        let fields = [
            self.regime.clone(),
            self.family.clone(),
            self.dtype.to_owned(),
            self.order.to_owned(),
            self.m.to_string(),
            self.n.to_string(),
            self.k.map(|k| k.to_string()).unwrap_or_default(),
            self.batch.to_string(),
            self.row.clone(),
            self.threads.to_string(),
            total,
            per_item,
            self.status.to_owned(),
            self.note.clone(),
            if self.samples_ms.is_empty() {
                String::new()
            } else {
                format!("{:?}", self.samples_ms)
            },
            self.rayon_threads.to_string(),
            self.vendor_threads
                .map(|n| n.to_string())
                .unwrap_or_default(),
            self.profile.clone(),
            milliseconds(self.timed_ms),
            milliseconds(self.check_ms),
        ];
        fields
            .iter()
            .map(|f| csv_field(f))
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The hardware profile the recorder declared, or empty: a hand-run harness has none, and an
/// empty column says so rather than naming a machine nothing declared.
fn profile_from_env() -> String {
    std::env::var("FBENCH_PROFILE").unwrap_or_default()
}

/// The parsed arguments with the recorder's profile attached. One function, so the wiring is
/// what the unit test calls instead of rebuilding it.
fn args_from_env(argv: &[String]) -> Result<Args, String> {
    let mut args = parse(argv)?;
    args.profile = profile_from_env();
    Ok(args)
}

/// The hidden `--cell-spec` payload: everything one child process needs to build and run its cell
/// batch. The parent writes it as `key=value;...` (no shell quoting, no JSON dependency) and the
/// child writes `COUNT`/`RESULT` lines back on stdout.
#[derive(Clone, Debug)]
struct CellSpec {
    id: u64,
    /// First variant this child runs (`0` unless the parent respawns after a kill).
    from: usize,
    /// How many variants the parent expects in total, known before construction.
    n: usize,
    /// The parent's pid: the child exits immediately if it is not its parent, and never leaves an
    /// orphan when the parent dies.
    parent_pid: u32,
    run: bool,
    row: String,
    family: String,
    dtype: Dtype,
    order: Order,
    case: Case,
    threads: usize,
    prime_ms: u64,
    reps: usize,
    limit_s: u64,
}

impl CellSpec {
    fn encode(&self) -> String {
        let k = self.case.k.map(|k| k.to_string()).unwrap_or_default();
        format!(
            "id={};from={};variants={};parent_pid={};run={};row={};family={};dtype={};order={};m={};column={};k={};batch={};threads={};prime_ms={};reps={};limit_s={}",
            self.id,
            self.from,
            self.n,
            self.parent_pid,
            self.run as u8,
            self.row,
            self.family,
            self.dtype.label(),
            self.order.label(),
            self.case.m,
            self.case.n,
            k,
            self.case.batch,
            self.threads,
            self.prime_ms,
            self.reps,
            self.limit_s,
        )
    }

    fn parse(text: &str) -> Result<CellSpec, String> {
        let get = |key: &str| -> Option<&str> {
            text.split(';')
                .find_map(|field| field.strip_prefix(key).and_then(|rest| rest.strip_prefix('=')))
        };
        let number = |key: &str| -> Result<u64, String> {
            get(key)
                .ok_or_else(|| format!("cell spec has no {key}"))?
                .parse()
                .map_err(|e| format!("cell spec {key}: {e}"))
        };
        let dtype = match get("dtype").ok_or("cell spec has no dtype")? {
            "f64" => Dtype::F64,
            "c64" => Dtype::C64,
            "f32" => Dtype::F32,
            "c32" => Dtype::C32,
            other => return Err(format!("cell spec dtype {other}")),
        };
        let order = match get("order").ok_or("cell spec has no order")? {
            "c" => Order::C,
            "f" => Order::F,
            other => return Err(format!("cell spec order {other}")),
        };
        let k = match get("k").unwrap_or_default() {
            "" => None,
            k => Some(k.parse().map_err(|e| format!("cell spec k: {e}"))?),
        };
        Ok(CellSpec {
            id: number("id")?,
            from: number("from")? as usize,
            n: number("variants")? as usize,
            parent_pid: number("parent_pid")? as u32,
            run: number("run")? != 0,
            row: get("row").ok_or("cell spec has no row")?.to_owned(),
            family: get("family").ok_or("cell spec has no family")?.to_owned(),
            dtype,
            order,
            case: Case {
                m: number("m")? as usize,
                n: number("column")? as usize,
                k,
                batch: number("batch")? as usize,
            },
            threads: number("threads")? as usize,
            prime_ms: number("prime_ms")?,
            reps: number("reps")? as usize,
            limit_s: number("limit_s")?,
        })
    }
}

/// Exit the child when its parent dies. The parent promises to kill the child at the deadline; if
/// the parent itself is killed (SIGKILL cannot be caught), a child left running would be an orphan
/// spending the measurement budget with nobody reading it. The parent's pid is checked on a
/// watchdog thread because `getppid` changes when the process is reparented.
#[cfg(unix)]
fn watch_parent(parent_pid: u32) {
    extern "C" {
        fn getppid() -> std::os::raw::c_int;
    }
    let expected = parent_pid as std::os::raw::c_int;
    if unsafe { getppid() } != expected {
        std::process::exit(137);
    }
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_millis(500));
        if unsafe { getppid() } != expected {
            std::process::exit(137);
        }
    });
}

#[cfg(not(unix))]
fn watch_parent(_parent_pid: u32) {}

fn child_main(spec_text: &str) -> ExitCode {
    let spec = match CellSpec::parse(spec_text) {
        Ok(spec) => spec,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    watch_parent(spec.parent_pid);
    if let Err(e) = configure_threads(spec.threads) {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    let limit = (spec.limit_s > 0).then(|| Duration::from_secs(spec.limit_s));
    let start = Instant::now();
    timing::arm_at(timing::deadline_at(start, limit));
    let built = ops_for(&spec.row, spec.dtype, &spec.family, spec.case, spec.order);
    let construction_cut = timing::expired();
    let construction = start.elapsed();
    let ops = match built {
        Ok(ops) => ops,
        Err(note) => {
            // The child is the only place that builds, so an unsupported family is reported here.
            println!("SKIP\t{note}");
            let _ = std::io::stdout().flush();
            return ExitCode::SUCCESS;
        }
    };
    println!("COUNT\t{}", ops.len());
    let _ = std::io::stdout().flush();
    if construction_cut {
        println!(
            "RESULT\t{}\ttimeout\ttimeout {}\t\t\t",
            spec.from, spec.limit_s
        );
        let _ = std::io::stdout().flush();
        return ExitCode::SUCCESS;
    }
    for (index, mut op) in ops.into_iter().enumerate() {
        if index < spec.from {
            continue;
        }
        let cell_start = Instant::now();
        let cell = run_cell(
            &mut op,
            spec.run,
            spec.threads,
            &spec.row,
            spec.prime_ms,
            spec.reps,
            Budget {
                limit_s: spec.limit_s,
                deadline: variant_deadline(cell_start, limit, construction),
                construction_cut: false,
            },
        );
        println!(
            "RESULT\t{index}\t{}\t{}\t{}\t{}\t{}",
            cell.status,
            cell.note,
            cell.timed_ms.map(|v| format!("{v}")).unwrap_or_default(),
            cell.check_ms.map(|v| format!("{v}")).unwrap_or_default(),
            cell.samples
                .iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(","),
        );
        let _ = std::io::stdout().flush();
    }
    ExitCode::SUCCESS
}

/// One row result reported by a child, before it becomes a [`Cell`].
struct ChildResult {
    status: &'static str,
    note: String,
    timed_ms: Option<f64>,
    check_ms: Option<f64>,
    samples: Vec<f64>,
}

/// How many tab-separated fields a `RESULT` line has: see the child's single `println!` for it.
const RESULT_FIELDS: usize = 7;
/// `SKIP<TAB>note`, the only other line the child writes to stdout.
const SKIP_FIELDS: usize = 2;
/// `COUNT<TAB>n`, the first line of a spawn: how many variants the build produced.
const COUNT_FIELDS: usize = 2;

/// `COUNT<TAB>n`. The parent does not need the number (it reads one `RESULT` per variant), but a line
/// that carries something else is broken output, and the harness does not record broken output as
/// success. A repeated, well-formed `COUNT` stays accepted: a flooding child must still be caught by
/// the deadline, not by the protocol.
fn parse_count(fields: &[&str]) -> Result<usize, String> {
    if fields.len() != COUNT_FIELDS {
        return Err(format!(
            "COUNT has {} field(s), expected {COUNT_FIELDS}",
            fields.len()
        ));
    }
    fields[1]
        .parse::<usize>()
        .map_err(|e| format!("COUNT payload is not a count: {e}"))
}

fn parse_number(field: &str, name: &str) -> Result<Option<f64>, String> {
    if field.is_empty() {
        return Ok(None);
    }
    let value: f64 = field
        .parse()
        .map_err(|e| format!("RESULT {name}: {e}"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("RESULT {name} is not a finite non-negative number: {field}"));
    }
    Ok(Some(value))
}

/// The index is the caller's variant number, so it is checked here, once, right after it is parsed:
/// an index a `usize` can hold but this build cannot (`18446744073709551615` from a broken child)
/// must be a failed row, not an arithmetic overflow in the caller's error message.
fn parse_result(fields: &[&str], n: usize) -> Result<(usize, ChildResult), String> {
    // The line has a fixed shape, written by the child in one place:
    // `RESULT<TAB>index<TAB>status<TAB>note<TAB>timed_ms<TAB>check_ms<TAB>samples`. A field more or
    // fewer is broken output: an extra one would carry arbitrary text past both the number checks
    // and the extra-output check, because it belongs to this line.
    if fields.len() != RESULT_FIELDS {
        return Err(format!(
            "RESULT has {} field(s), expected {RESULT_FIELDS}",
            fields.len()
        ));
    }
    let index_text = fields.get(1).ok_or("RESULT without an index")?;
    let index: usize = index_text
        .parse()
        .map_err(|e| format!("RESULT index: {e}"))?;
    if index >= n {
        return Err(format!(
            "RESULT index {index_text} is out of range; this build has {n} variant(s)"
        ));
    }
    let status = match *fields.get(2).ok_or("RESULT without a status")? {
        "ok" => "ok",
        "timeout" => "timeout",
        "failed" => "failed",
        other => return Err(format!("RESULT has an unknown status {other:?}")),
    };
    let timed_ms = parse_number(fields.get(4).unwrap_or(&""), "timed_ms")?;
    let check_ms = parse_number(fields.get(5).unwrap_or(&""), "check_ms")?;
    let samples_field = fields.get(6).unwrap_or(&"");
    let samples = if samples_field.is_empty() {
        Vec::new()
    } else {
        let mut samples = Vec::new();
        for value in samples_field.split(',') {
            let parsed: f64 = value
                .parse()
                .map_err(|e| format!("RESULT sample {value:?}: {e}"))?;
            if !parsed.is_finite() || parsed < 0.0 {
                return Err(format!(
                    "RESULT sample is not a finite non-negative number: {value}"
                ));
            }
            samples.push(parsed);
        }
        samples
    };
    Ok((
        index,
        ChildResult {
            status,
            note: fields.get(3).unwrap_or(&"").to_string(),
            timed_ms,
            check_ms,
            samples,
        },
    ))
}

/// A result is only `ok` if it has the shape an `ok` row promises: finite timed/check times and,
/// in run mode, one sample per repetition. Anything else goes down the failed-cell path.
fn validate_result(result: &ChildResult, run: bool, reps: usize) -> Result<(), String> {
    if result.status != "ok" {
        return Ok(());
    }
    if result.timed_ms.is_none() || result.check_ms.is_none() {
        return Err("RESULT ok without finite timed_ms and check_ms".to_owned());
    }
    if run && result.samples.len() != reps {
        return Err(format!(
            "RESULT ok with {} sample(s), expected {reps}",
            result.samples.len()
        ));
    }
    Ok(())
}

/// Run one `build_ops` batch in a child process and enforce each cell's limit by killing it.
///
/// The parent does not build and does not measure: it spawns the child, reads `COUNT` (how many
/// variants the build produced) and one `RESULT` per variant, writes the row for each, and kills
/// the child when the running variant's deadline passes. A killed child is `timeout`, `CELL done
/// ... cut=timeout` closes it, and the next variant starts a new deadline. Cooperative polls stay
/// in the child as an early-exit optimisation; the guarantee is here.
/// How many rows one `ops_for` build produces, known without building. The parent needs it before
/// construction: a build cut during construction must still leave one terminal row per variant.
fn variant_count(row: &str, family: &str) -> usize {
    // The fake child used by the transition tests cannot build an `Op`; the parent's variant count
    // is overridden for it. Production never sets either variable.
    if std::env::var_os("FBENCH_CELL_CHILD").is_some() {
        if let Some(n) = std::env::var("FBENCH_TEST_VARIANTS")
            .ok()
            .and_then(|value| value.parse().ok())
        {
            return n;
        }
    }
    if !row.starts_with("faer-") {
        return 1;
    }
    match family {
        "lu" | "full_piv_lu" | "cholesky" | "solve" | "svd_values" | "svd_thin" | "eigh"
        | "householder_apply" => 2,
        _ => 1,
    }
}

/// Reap a child without an unbounded wait: `try_wait` first, then `SIGKILL` and wait. A child that
/// closed stdout but kept running must not hold the run open.
/// How long a reap may take before the child is abandoned. There is no blocking `wait()`: a
/// process that cannot be killed (uninterruptible I/O) would otherwise hold the run forever.
const REAP_BOUND: Duration = Duration::from_secs(5);
/// How long a child that has nothing more to report gets to exit on its own before it is killed.
/// This is what distinguishes "reported its last result, then exited 3" from "still running".
const REAP_GRACE: Duration = Duration::from_millis(250);

enum Reaped {
    /// The child exited on its own.
    Exited(std::process::ExitStatus),
    /// The parent killed it because it had nothing more to report (or was over its limit).
    Killed(std::process::ExitStatus),
    TimedOut { kill_error: Option<String> },
}

impl Reaped {
    fn success(&self) -> bool {
        matches!(self, Reaped::Exited(status) if status.success())
    }

    fn text(&self) -> String {
        match self {
            Reaped::Exited(status) | Reaped::Killed(status) => status.to_string(),
            Reaped::TimedOut { kill_error } => match kill_error {
                Some(e) => format!("reap timed out; kill failed: {e}"),
                None => "reap timed out".to_owned(),
            },
        }
    }
}

fn try_wait_until(child: &mut std::process::Child, bound: Duration) -> Option<std::process::ExitStatus> {
    let deadline = Instant::now() + bound;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Some(status),
            Ok(None) => {}
            Err(_) => return None,
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Reap a child that should be finished (stdout closed, or the last result received). It gets
/// [`REAP_GRACE`] to exit on its own, so a real exit status is seen; only then is it killed.
fn reap_child(mut child: std::process::Child) -> Reaped {
    if let Some(status) = try_wait_until(&mut child, REAP_GRACE) {
        return Reaped::Exited(status);
    }
    let kill_error = child.kill().err().map(|e| e.to_string());
    if let Some(status) = try_wait_until(&mut child, REAP_BOUND) {
        return Reaped::Killed(status);
    }
    // The child did not die inside the bound. Abandon it rather than block; the caller records a
    // failed row and logs the timeout. (SIGKILL is uninterruptible, so this is the pathological case.)
    Reaped::TimedOut { kill_error }
}

/// Reap at a deadline: kill now, then wait a bounded time for the signal to land.
fn reap_now(mut child: std::process::Child) -> Reaped {
    if let Ok(Some(status)) = child.try_wait() {
        return Reaped::Exited(status);
    }
    let kill_error = child.kill().err().map(|e| e.to_string());
    if let Some(status) = try_wait_until(&mut child, REAP_BOUND) {
        return Reaped::Killed(status);
    }
    Reaped::TimedOut { kill_error }
}

fn unix_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

struct CellChild {
    child: std::process::Child,
    pid: u32,
    rx: std::sync::mpsc::Receiver<Result<String, String>>,
    reader: std::thread::JoinHandle<()>,
}

fn spawn_cell(exe: &std::path::Path, spec: &CellSpec) -> Result<CellChild, String> {
    use std::sync::mpsc;
    let mut child = Command::new(exe)
        .arg("--cell-spec")
        .arg(spec.encode())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("cannot start the cell child: {e}"))?;
    let pid = child.id();
    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel::<Result<String, String>>();
    let reader = std::thread::spawn(move || {
        // `lines()` turns one unreadable byte into the end of the stream, and a parent cannot tell
        // that from a child with nothing more to say. Read whole lines and report a line that is
        // not text as broken output, so it fails the variant instead of ending or passing it.
        // The line ending follows `lines()`: drop one `\n` and, only then, one `\r`. A last line
        // with no newline keeps its carriage return.
        let mut reader = BufReader::new(stdout);
        loop {
            let mut raw = Vec::new();
            match reader.read_until(b'\n', &mut raw) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
            if raw.last() == Some(&b'\n') {
                raw.pop();
                if raw.last() == Some(&b'\r') {
                    raw.pop();
                }
            }
            let line = String::from_utf8(raw)
                .map_err(|e| format!("the cell child wrote a line that is not text: {e}"));
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    Ok(CellChild {
        child,
        pid,
        rx,
        reader,
    })
}

enum Wake {
    Line(String),
    /// A line of the child's output that is not text: broken output, not an end of the stream.
    Broken(String),
    Timeout,
    Closed,
}

/// Read what a child still says into `extra` (line count, first line) until its stdout closes or
/// `deadline` passes. Used after the last expected result: anything the child says there is output
/// the protocol does not allow.
fn drain_extra(
    rx: &std::sync::mpsc::Receiver<Result<String, String>>,
    deadline: Instant,
    extra: &mut Option<(usize, String)>,
) {
    loop {
        let line = match wait_line(rx, Some(deadline)) {
            Wake::Line(line) => line,
            Wake::Broken(e) => format!("<{e}>"),
            Wake::Timeout | Wake::Closed => return,
        };
        match extra {
            Some((count, _)) => *count += 1,
            None => *extra = Some((1, line)),
        }
    }
}

fn wait_line(
    rx: &std::sync::mpsc::Receiver<Result<String, String>>,
    deadline: Option<Instant>,
) -> Wake {
    use std::sync::mpsc::RecvTimeoutError;
    // `recv_timeout` returns an already-queued message before its deadline, so a child that floods
    // COUNT lines would never reach the timeout branch. Check the clock before receiving.
    if let Some(deadline) = deadline {
        if Instant::now() >= deadline {
            return Wake::Timeout;
        }
    }
    match deadline {
        Some(deadline) => {
            let remaining = deadline.saturating_duration_since(Instant::now());
            match rx.recv_timeout(remaining) {
                Ok(Ok(line)) => Wake::Line(line),
                Ok(Err(e)) => Wake::Broken(e),
                Err(RecvTimeoutError::Timeout) => Wake::Timeout,
                Err(RecvTimeoutError::Disconnected) => Wake::Closed,
            }
        }
        None => match rx.recv() {
            Ok(Ok(line)) => Wake::Line(line),
            Ok(Err(e)) => Wake::Broken(e),
            Err(_) => Wake::Closed,
        },
    }
}

/// Run one build in a child process, one terminal row per variant.
///
/// Invariant: each variant gets exactly one `CELL start`, exactly one terminal row (`ok`,
/// `timeout`/`cut=timeout`, or `failed`) and one `CELL done`. A killed variant is the only one cut:
/// the parent respawns a child for the remaining variants (`from=`), so they actually run. The
/// variant count is fixed before construction (`variant_count`), so a construction kill still
/// leaves a row for every variant it did not reach.
#[allow(clippy::too_many_arguments)]
fn run_build_in_child(
    args: &Args,
    csv: &mut Csv,
    threads: usize,
    row: &str,
    dtype: &Dtype,
    family: &str,
    case: Case,
    order: Order,
    run: bool,
    limit: Option<Duration>,
    all_ok: &mut bool,
) -> Result<(), String> {
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_CELL: AtomicU64 = AtomicU64::new(0);
    let id = NEXT_CELL.fetch_add(1, Ordering::Relaxed) + 1;
    let n = variant_count(row, family);
    let parent_pid = std::process::id();
    let exe = std::env::var_os("FBENCH_CELL_CHILD")
        .map(std::path::PathBuf::from)
        .map(Ok)
        .unwrap_or_else(std::env::current_exe)
        .map_err(|e| format!("cannot find this binary: {e}"))?;
    let spec_at = |from: usize| CellSpec {
        id,
        from,
        n,
        parent_pid,
        run,
        row: row.to_owned(),
        family: family.to_owned(),
        dtype: *dtype,
        order,
        case,
        threads,
        prime_ms: args.prime_ms,
        reps: args.reps,
        limit_s: args.cell_timeout,
    };

    let mut next = 0usize;
    let mut current: Option<CellChild> = None;
    // Whether the child currently running has produced a result in this spawn. A child that exits
    // after a result (for example the construction-cut child, which reports one `timeout` and
    // stops) has not failed the next variant: the next variant is respawned.
    let mut completed_in_child = false;
    // The last variant's result is held until the child's exit is known: a child that reports the
    // last result and then crashes must make that row failed, not successful.
    let mut pending_final: Option<Cell> = None;
    // The last result this child reported. The cooperative construction cut is exactly
    // "a timeout result, then a successful exit"; any other exit after a result is a crash.
    let mut last_result_timeout = false;
    let mut spawn_time = Instant::now();
    let mut construction = Duration::ZERO;
    let mut deadline = variant_deadline(spawn_time, limit, construction);

    let mut finish = |csv: &mut Csv, cell: Cell, index: usize| {
        let cell = match cell.status {
            "ok" => match derived_ok_values_are_finite(&cell, case.batch, args.statistic) {
                Ok(()) => cell,
                Err(note) => Cell {
                    status: "failed",
                    note,
                    samples: Vec::new(),
                    timed_ms: None,
                    check_ms: None,
                },
            },
            _ => cell,
        };
        heartbeat(
            "done",
            row,
            family,
            dtype,
            order,
            case,
            format!(
                " id={id} variant={} status={}{}",
                index + 1,
                cell.status,
                if cell.status == "timeout" {
                    " cut=timeout"
                } else {
                    ""
                }
            ),
        );
        if !run {
            let label = format!(
                "{row} {} {} {} {}",
                family,
                dtype.label(),
                order.label(),
                case.shape()
            );
            match cell.status {
                "ok" => println!("ok {label} {}", cell.note),
                "timeout" => println!("TIMEOUT {label} {}", cell.note),
                // A provider that cannot take the family is a skip, not a wrong answer.
                "skipped" => println!("skip {label} {}", cell.note),
                _ => println!("MISMATCH {label} {}", cell.note),
            }
        }
        if cell.status != "ok" && cell.status != "skipped" {
            *all_ok = false;
        }
        emit(csv, args, threads, dtype, family, case, order, row, cell);
    };
    let start = |index: usize| {
        heartbeat(
            "start",
            row,
            family,
            dtype,
            order,
            case,
            format!(" id={id} variant={}/{n}", index + 1),
        );
    };
    // Anything the child says that the protocol does not allow closes the variant that was running:
    // the caller writes this failed row and respawns the next variant. `pid` and `status` name the
    // child that said it and how it ended.
    let protocol_error = |index: usize, error: String, pid: u32, status: String| -> Cell {
        heartbeat(
            "protocol-error",
            row,
            family,
            dtype,
            order,
            case,
            format!(
                " id={id} pid={pid} variant={}/{n} error={error} status={status}",
                index + 1
            ),
        );
        Cell {
            status: "failed",
            note: error,
            samples: Vec::new(),
            timed_ms: None,
            check_ms: None,
        }
    };

    if n == 0 {
        return Ok(());
    }
    start(0);

    while next < n {
        let handle = match current.take() {
            Some(handle) => handle,
            None => {
                spawn_time = Instant::now();
                construction = Duration::ZERO;
                completed_in_child = false;
                last_result_timeout = false;
                let spec = spec_at(next);
                let handle = match spawn_cell(&exe, &spec) {
                    Ok(handle) => handle,
                    Err(e) => {
                        // A missing binary or process-resource exhaustion is a failure of this
                        // cell, not a reason to leave the started variant without a terminal row.
                        // Close it as `failed` and try the next variant (which may fail too).
                        heartbeat(
                            "spawn-failed",
                            row,
                            family,
                            dtype,
                            order,
                            case,
                            format!(" id={id} variant={}/{n} error={e}", next + 1),
                        );
                        finish(
                            csv,
                            Cell {
                                status: "failed",
                                note: format!("could not start the cell child: {e}"),
                                samples: Vec::new(),
                                timed_ms: None,
                                check_ms: None,
                            },
                            next,
                        );
                        next += 1;
                        if next < n {
                            start(next);
                        }
                        continue;
                    }
                };
                heartbeat(
                    "build",
                    row,
                    family,
                    dtype,
                    order,
                    case,
                    format!(" id={id} pid={} from={}", handle.pid, next + 1),
                );
                deadline = variant_deadline(spawn_time, limit, construction);
                handle
            }
        };

        match wait_line(&handle.rx, deadline) {
            Wake::Line(line) => {
                let fields: Vec<&str> = line.split('\t').collect();
                match fields.first().copied() {
                    Some("COUNT") => {
                        if let Err(e) = parse_count(&fields) {
                            let pid = handle.pid;
                            let reaped = reap_child(handle.child);
                            drop(handle.reader);
                            let cell = protocol_error(next, e, pid, reaped.text());
                            finish(csv, cell, next);
                            next += 1;
                            current = None;
                            if next < n {
                                start(next);
                            }
                            continue;
                        }
                        construction = spawn_time.elapsed();
                    }
                    Some("SKIP") => {
                        if fields.len() != SKIP_FIELDS {
                            // A line whose shape the protocol does not allow is broken output here
                            // too: a `SKIP` with a stray field cannot become a `skipped` row.
                            let pid = handle.pid;
                            let reaped = reap_child(handle.child);
                            drop(handle.reader);
                            let error =
                                format!("SKIP has {} field(s), expected {SKIP_FIELDS}", fields.len());
                            let cell = protocol_error(next, error, pid, reaped.text());
                            finish(csv, cell, next);
                            next += 1;
                            current = None;
                            if next < n {
                                start(next);
                            }
                            continue;
                        }
                        let note = fields.get(1).unwrap_or(&"").to_string();
                        let pid = handle.pid;
                        // Output after a SKIP is broken output too: the row is only `skipped` if
                        // the child said nothing more, before or during its exit.
                        let mut extra: Option<(usize, String)> = None;
                        drain_extra(&handle.rx, Instant::now() + REAP_GRACE, &mut extra);
                        let reaped = reap_child(handle.child);
                        drain_extra(&handle.rx, Instant::now() + REAP_GRACE, &mut extra);
                        drop(handle.reader);
                        current = None;
                        if !completed_in_child {
                            // A genuine build-level skip: settle it only after the reap, and only if
                            // nothing followed it.
                            if let Some((count, first)) = &extra {
                                let cell = protocol_error(
                                    0,
                                    format!(
                                        "the cell child reported a skip and then {count} extra line(s), starting with {first:?}"
                                    ),
                                    pid,
                                    reaped.text(),
                                );
                                finish(csv, cell, 0);
                            } else if reaped.success() {
                                finish(csv, Cell::skipped(note), 0);
                            } else {
                                let status = reaped.text();
                                heartbeat(
                                    "skip-failed",
                                    row,
                                    family,
                                    dtype,
                                    order,
                                    case,
                                    format!(" id={id} variant=1/{n} status={status}"),
                                );
                                finish(
                                    csv,
                                    Cell {
                                        status: "failed",
                                        note: format!(
                                            "the cell child reported a skip and then exited with {status}: {note}"
                                        ),
                                        samples: Vec::new(),
                                        timed_ms: None,
                                        check_ms: None,
                                    },
                                    0,
                                );
                            }
                            return Ok(());
                        }
                        if next >= n {
                            // Every variant already reported; the late SKIP is extra output. Settle
                            // the held last row by the exit status and stop.
                            if let Some(cell) = pending_final.take() {
                                let cell = match &reaped {
                                    Reaped::Exited(status) if !status.success() => Cell {
                                        status: "failed",
                                        note: format!(
                                            "the cell child exited with {status} after reporting its last result"
                                        ),
                                        samples: Vec::new(),
                                        timed_ms: None,
                                        check_ms: None,
                                    },
                                    _ => cell,
                                };
                                finish(csv, cell, n - 1);
                            }
                            return Ok(());
                        }
                        // A SKIP after a result is a protocol error for the pending variant; a
                        // replacement child (from=) runs the rest.
                        let status = reaped.text();
                        heartbeat(
                            "late-skip",
                            row,
                            family,
                            dtype,
                            order,
                            case,
                            format!(
                                " id={id} variant={}/{n} status={status} note={note}",
                                next + 1
                            ),
                        );
                        finish(
                            csv,
                            Cell {
                                status: "failed",
                                note: format!("the cell child sent SKIP after a result: {note}"),
                                samples: Vec::new(),
                                timed_ms: None,
                                check_ms: None,
                            },
                            next,
                        );
                        next += 1;
                        if next < n {
                            start(next);
                        }
                        continue;
                    }
                    Some("RESULT") => {
                        let parsed = parse_result(&fields, n).and_then(|(result_index, result)| {
                            if result_index != next {
                                return Err(format!(
                                    "the cell child reported variant {} while {} was running",
                                    result_index + 1,
                                    next + 1
                                ));
                            }
                            validate_result(&result, run, args.reps)?;
                            Ok(result)
                        });
                        let result = match parsed {
                            Ok(result) => result,
                            Err(e) => {
                                let pid = handle.pid;
                                let reaped = reap_child(handle.child);
                                drop(handle.reader);
                                let cell = protocol_error(next, e, pid, reaped.text());
                                finish(csv, cell, next);
                                next += 1;
                                current = None;
                                if next < n {
                                    start(next);
                                }
                                continue;
                            }
                        };
                        let cell = Cell {
                            status: result.status,
                            note: result.note,
                            samples: result.samples,
                            timed_ms: result.timed_ms,
                            check_ms: result.check_ms,
                        };
                        last_result_timeout = cell.status == "timeout";
                        completed_in_child = true;
                        if next + 1 == n {
                            pending_final = Some(cell);
                            next += 1;
                            deadline = None;
                        } else {
                            finish(csv, cell, next);
                            next += 1;
                            start(next);
                            deadline = variant_deadline(Instant::now(), limit, construction);
                        }
                    }
                    other => {
                        // A line the protocol does not define is broken output wherever it appears.
                        // Dropping it here made acceptance depend on position: the same line was
                        // ignored before a result and rejected after the last one.
                        let pid = handle.pid;
                        let reaped = reap_child(handle.child);
                        drop(handle.reader);
                        let error = format!(
                            "the cell child wrote a line that is not part of the protocol: {:?}",
                            other.unwrap_or(&line)
                        );
                        let cell = protocol_error(next, error, pid, reaped.text());
                        finish(csv, cell, next);
                        next += 1;
                        current = None;
                        if next < n {
                            start(next);
                        }
                        continue;
                    }
                }
                current = Some(handle);
            }
            Wake::Broken(e) => {
                // Bytes the protocol cannot carry are broken output for the variant that is
                // running: a failed row, not a silent end of the stream.
                let pid = handle.pid;
                let reaped = reap_child(handle.child);
                drop(handle.reader);
                let cell = protocol_error(next, e, pid, reaped.text());
                finish(csv, cell, next);
                next += 1;
                current = None;
                if next < n {
                    start(next);
                }
            }
            Wake::Timeout => {
                let killed_at = unix_millis();
                let pid = handle.pid;
                let reaped = reap_now(handle.child);
                let status = reaped.text();
                let reaped_at = unix_millis();
                drop(handle.reader);
                heartbeat(
                    "kill",
                    row,
                    family,
                    dtype,
                    order,
                    case,
                    format!(
                        " id={id} pid={pid} variant={variant}/{n} signal=SIGKILL at_ms={killed_at} reaped_at={reaped_at} status={status}",
                        variant = next + 1
                    ),
                );
                finish(csv, Cell::cut(args.cell_timeout), next);
                next += 1;
                current = None;
                if next < n {
                    start(next);
                }
            }
            Wake::Closed => {
                let pid = handle.pid;
                let reaped = reap_child(handle.child);
                let status = reaped.text();
                let cooperative = completed_in_child && reaped.success() && last_result_timeout;
                drop(handle.reader);
                heartbeat(
                    "exit",
                    row,
                    family,
                    dtype,
                    order,
                    case,
                    format!(
                        " id={id} pid={pid} status={status} cooperative={cooperative} at_ms={}",
                        unix_millis()
                    ),
                );
                current = None;
                if cooperative {
                    // The construction-cut child reports exactly one timeout and exits successfully.
                    // It did not fail the next variant: respawn it, without a new start.
                    continue;
                }
                let cell = Cell {
                    status: "failed",
                    note: format!("the cell child exited with {status} before its result"),
                    samples: Vec::new(),
                    timed_ms: None,
                    check_ms: None,
                };
                finish(csv, cell, next);
                next += 1;
                if next < n {
                    start(next);
                }
            }
        }
    }

    if let Some(handle) = current {
        // Every variant has a result, but the child may still be talking: read it out to the end
        // (bounded) before reaping. Finishing with `next == n` used to end the loop here, so a
        // child that reported its last result and then printed anything else was accepted as `ok`.
        let mut extra: Option<(usize, String)> = None;
        drain_extra(&handle.rx, Instant::now() + REAP_GRACE, &mut extra);
        // The child has nothing left to say (or the bound passed). Reap it (kill if it is still
        // running) and only then settle the held last row.
        let reaped = reap_child(handle.child);
        // Output that landed while the child was exiting is queued behind the reap, and stdout can
        // close after it: read it too, so "printed garbage, then exited 0" is not a clean row.
        drain_extra(&handle.rx, Instant::now() + REAP_GRACE, &mut extra);
        drop(handle.reader);
        if let Some(cell) = pending_final.take() {
            let cell = match &extra {
                Some((count, first)) => {
                    heartbeat(
                        "extra-output",
                        row,
                        family,
                        dtype,
                        order,
                        case,
                        format!(
                            " id={id} variant={n}/{n} lines={count} first={first:?} status={}",
                            reaped.text()
                        ),
                    );
                    Cell {
                        status: "failed",
                        note: format!(
                            "the cell child reported {count} extra line(s) after its last result, starting with {first:?}"
                        ),
                        samples: Vec::new(),
                        timed_ms: None,
                        check_ms: None,
                    }
                }
                None => match &reaped {
                    Reaped::Exited(status) if !status.success() => Cell {
                        status: "failed",
                        note: format!(
                            "the cell child exited with {status} after reporting its last result"
                        ),
                        samples: Vec::new(),
                        timed_ms: None,
                        check_ms: None,
                    },
                    // The parent had to kill the child, or could not reap it at all. Either way the
                    // last cell never finished, so its reported row is not a measurement: acceptance
                    // ③ (a final cell that dies must fail the run) covers this, not only a non-zero
                    // exit the parent did not cause.
                    Reaped::Killed(_) | Reaped::TimedOut { .. } => {
                        heartbeat(
                            "final-reap",
                            row,
                            family,
                            dtype,
                            order,
                            case,
                            format!(" id={id} variant={}/{n} status={}", n, reaped.text()),
                        );
                        Cell {
                            status: "failed",
                            note: format!(
                                "the final cell did not complete: the child did not exit after reporting its last result ({})",
                                reaped.text()
                            ),
                            samples: Vec::new(),
                            timed_ms: None,
                            check_ms: None,
                        }
                    }
                    // `Reaped::Exited` with a success status: the child reported its last result,
                    // said nothing more, and exited cleanly.
                    _ => cell,
                },
            };
            finish(csv, cell, n - 1);
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    // Hidden child mode: one batch, measured in its own process so the parent can kill it. It is
    // not part of the recorder's CLI.
    if argv.get(1).map(String::as_str) == Some("--cell-spec") {
        return match argv.get(2) {
            Some(spec) => child_main(spec),
            None => {
                eprintln!("--cell-spec needs a payload");
                ExitCode::from(2)
            }
        };
    }
    let args = match args_from_env(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if let Some(n) = args.threads {
        if let Err(e) = configure_threads(n) {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    }
    if args.command == "info" {
        info(args.threads);
        return ExitCode::SUCCESS;
    }
    let threads = args.threads.expect("verify and run require --threads");

    let mut rows = engine_rows();
    if !args.rows.is_empty() {
        rows.retain(|row| args.rows.contains(row));
    }
    let n_rows = rows.len();
    if n_rows > 0 {
        rows.rotate_left(args.rotate_rows % n_rows);
    }
    let orders = [Order::C, Order::F];
    let run = args.command == "run";
    let mut csv = match Csv::open(args.csv.as_ref()) {
        Ok(csv) => csv,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let mut all_ok = true;
    let limit = (args.cell_timeout > 0).then(|| Duration::from_secs(args.cell_timeout));

    for dtype in &args.dtypes {
        for family in &args.families {
            for shape in &args.shapes {
                for batch in &args.batches {
                    let case = match shape.case(family, *batch) {
                        Ok(case) => case,
                        Err(note) => {
                            for order in orders {
                                for row in &rows {
                                    if !run {
                                        println!(
                                            "skip {row} {} {} {} {} {note}",
                                            family,
                                            dtype.label(),
                                            order.label(),
                                            case_for_note(*shape, family, *batch).shape()
                                        );
                                    }
                                    emit(
                                        &mut csv,
                                        &args,
                                        threads,
                                        dtype,
                                        family,
                                        case_for_note(*shape, family, *batch),
                                        order,
                                        row,
                                        Cell::skipped(note.clone()),
                                    );
                                }
                            }
                            continue;
                        }
                    };
                    for order in orders {
                        for row in &rows {
                            if let Err(e) = require_threads(threads) {
                                eprintln!("{e}");
                                return ExitCode::from(2);
                            }
                            if let Err(e) = run_build_in_child(
                                &args,
                                &mut csv,
                                threads,
                                row,
                                dtype,
                                family,
                                case,
                                order,
                                run,
                                limit,
                                &mut all_ok,
                            ) {
                                eprintln!("{e}");
                                return ExitCode::from(2);
                            }
                        }
                    }
                }
            }
        }
    }

    if !run && all_ok {
        println!("all comparisons within tolerance");
    }

    exit_code(all_ok, csv.failed)
}

/// The process's exit status: a CSV that lost a row is 2 -- the recorder reads a non-zero exit, and
/// a report missing a cell is the failure this change exists to remove -- a cell that was not `ok`
/// is 1, and everything passed is 0.
fn exit_code(all_ok: bool, csv_failed: bool) -> ExitCode {
    if csv_failed {
        return ExitCode::from(2);
    }
    if all_ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// The case used only to name a shape in a skipped row when the shape itself
/// could not be turned into a case.
fn case_for_note(shape: RawShape, family: &str, batch: usize) -> Case {
    shape.case(family, batch).unwrap_or(Case {
        m: shape.m,
        n: shape.n,
        k: shape.k,
        batch,
    })
}

/// The CSV as it is produced: the header when the file is opened, then one row per cell, written
/// and flushed as that cell finishes.
///
/// The rows used to be buffered in memory and written once at the end, so a cell that never
/// finished cost the run every row it had already earned: after the 27.6 h stall there was no CSV
/// at all, and "still measuring" and "stuck" looked the same from outside. With the rows on disk
/// as they complete, the file itself is the progress record. Rows are appended in completion order,
/// which is the order the loop runs the cells in.
struct Csv {
    out: Option<(std::path::PathBuf, std::fs::File)>,
    /// Set when a row could not be written. A row that did not reach the file is a cell the report
    /// will not see, so the run must not exit successfully because of it.
    failed: bool,
}

const CSV_HEADER: &str = "regime,family,dtype,order,m,n,k,batch,row,threads,total_ms,per_item_us,status,note,samples_ms,rayon_threads,vendor_threads,profile,timed_ms,check_ms\n";

impl Csv {
    /// Open the run's CSV and write its header, or write nothing when no path was given (`verify`
    /// without `--csv`).
    fn open(path: Option<&String>) -> Result<Csv, String> {
        let out = match path {
            None => None,
            Some(path) => {
                use std::io::Write;
                let mut file =
                    std::fs::File::create(path).map_err(|e| format!("cannot write {path}: {e}"))?;
                file.write_all(CSV_HEADER.as_bytes())
                    .and_then(|()| file.flush())
                    .map_err(|e| format!("cannot write {path}: {e}"))?;
                Some((std::path::PathBuf::from(path), file))
            }
        };
        Ok(Csv { out, failed: false })
    }

    fn push(&mut self, record: Record) {
        use std::io::Write;
        let Some((path, file)) = &mut self.out else {
            return;
        };
        let mut line = record.line();
        line.push('\n');
        if let Err(e) = file.write_all(line.as_bytes()).and_then(|()| file.flush()) {
            eprintln!("cannot write {}: {e}", path.display());
            self.failed = true;
        }
    }
}

/// One line per cell on stderr, so a log that stops moving is a hang and not a guess. stdout is
/// what the recorder parses (`MISMATCH`, `all comparisons within tolerance`), so it is not used.
fn heartbeat(
    what: &str,
    row: &str,
    family: &str,
    dtype: &Dtype,
    order: Order,
    case: Case,
    tail: String,
) {
    eprintln!(
        "CELL {what} {row} {family} {} {} {}{tail}",
        dtype.label(),
        order.label(),
        case.shape()
    );
}

#[allow(clippy::too_many_arguments)]
fn emit(
    csv: &mut Csv,
    args: &Args,
    threads: usize,
    dtype: &Dtype,
    family: &str,
    case: Case,
    order: Order,
    row: &str,
    cell: Cell,
) {
    csv.push(Record {
        regime: args.regime.clone(),
        family: family.to_owned(),
        dtype: dtype.label(),
        order: order.label(),
        m: case.m,
        n: case.n,
        k: case.k,
        batch: case.batch,
        row: row.to_owned(),
        threads,
        rayon_threads: rayon::current_num_threads(),
        vendor_threads: vendor::reported_threads(),
        total_ms: summarize(&cell.samples, args.statistic),
        samples_ms: cell.samples,
        status: cell.status,
        note: cell.note,
        profile: args.profile.clone(),
        timed_ms: cell.timed_ms,
        check_ms: cell.check_ms,
    });
}

fn milliseconds(ms: Option<f64>) -> String {
    ms.map(|ms| format!("{ms:.6}")).unwrap_or_default()
}

/// What one cell produced: the row it gets, and the two times that used to be one number.
struct Cell {
    status: &'static str,
    note: String,
    /// The timed calls, in execution order. Empty when the cell did not measure anything.
    samples: Vec<f64>,
    timed_ms: Option<f64>,
    check_ms: Option<f64>,
}

impl Cell {
    fn skipped(note: String) -> Cell {
        Cell {
            status: "skipped",
            note,
            samples: Vec::new(),
            timed_ms: None,
            check_ms: None,
        }
    }

    /// A cell the harness cut off at its limit, naming the limit. The status says `timeout` rather
    /// than `failed` because a run that lost a cell to its limit and a run that caught a wrong
    /// answer are not the same event, and a cut cell is not an absent row. Nothing the cell then
    /// reported is kept: the reference math it abandoned mid-product can only produce a residual
    /// that is not evidence about the answer.
    fn cut(limit_s: u64) -> Cell {
        Cell {
            status: "timeout",
            note: format!("timeout {limit_s}"),
            samples: Vec::new(),
            // The region it was cut in did not finish, so it has no time: a partial number here
            // would be read as a measurement.
            timed_ms: None,
            check_ms: None,
        }
    }

    /// A cell that stopped: cut off by the cell clock, or failed on its own.
    fn stopped(limit_s: u64, detail: String) -> Cell {
        Cell::stopped_with(limit_s, timing::cut(), detail)
    }

    /// As [`Cell::stopped`], with the cut flag passed in. A cell that ran on a team member or a
    /// worker thread set `CUT` there, not on this thread, so the caller hands over the flag it read
    /// on the thread that ran the cell.
    fn stopped_with(limit_s: u64, cut: bool, detail: String) -> Cell {
        if cut {
            Cell::cut(limit_s)
        } else {
            Cell {
                status: "failed",
                note: detail,
                samples: Vec::new(),
                timed_ms: None,
                check_ms: None,
            }
        }
    }
}

/// The cell's own note and the reason it stopped, joined the way a failing row has always been
/// written (`<note>; <reason>`), or as just the reason when the cell has no note.
fn stopped_note(note: &str, reason: &str) -> String {
    if note.is_empty() {
        reason.to_owned()
    } else {
        format!("{note}; {reason}")
    }
}

/// One variant deadline: its own clock, less the construction every variant paid.
/// The loop used to share one deadline across every op `build_ops` produced, so a slow earlier
/// variant spent the budget of the ones after it.
fn variant_deadline(
    cell_start: Instant,
    limit: Option<Duration>,
    construction: Duration,
) -> Option<Instant> {
    timing::deadline_at(
        cell_start,
        limit.map(|limit| limit.saturating_sub(construction)),
    )
}

#[cfg(test)]
mod variant_budget_tests {
    use super::*;

    #[test]
    fn two_variants_do_not_share_one_deadline() {
        let limit = Some(Duration::from_secs(1));
        let construction = Duration::from_millis(100);
        let start = Instant::now();
        let first = variant_deadline(start, limit, construction).unwrap();
        // The second variant starts after the first has used 600 ms of wall time.
        let second_start = start + Duration::from_millis(600);
        let second = variant_deadline(second_start, limit, construction).unwrap();
        assert_eq!(
            second.duration_since(second_start),
            Duration::from_millis(900),
            "each variant gets the cell budget, less construction, from its own start"
        );
        assert!(
            first < second,
            "one shared deadline would leave the second variant 300 ms"
        );
    }

    #[test]
    fn an_unrepresentable_limit_is_still_a_limit() {
        assert!(crate::timing::deadline_in(Some(Duration::from_secs(u64::MAX))).is_some());
    }
}

/// One cell's budget: the configured limit (the note names it), and whether the reference
/// construction that produced these ops already crossed it.
#[derive(Clone, Copy)]
struct Budget {
    limit_s: u64,
    /// The absolute deadline, fixed when the cell (including its construction) started.
    deadline: Option<Instant>,
    construction_cut: bool,
}

impl Budget {
    /// A budget for a cell whose ops are built by the caller, with nothing yet spent. Tests use
    /// this; the executor fixes its deadline before construction instead.
    #[cfg(test)]
    fn new(limit_s: u64) -> Budget {
        Budget {
            limit_s,
            deadline: timing::deadline_in((limit_s > 0).then(|| Duration::from_secs(limit_s))),
            construction_cut: false,
        }
    }
}

/// Run one cell under its own clock: the timed repetitions and the check that follows them, or one
/// verification call and its check. The limit covers the whole cell and is disarmed afterwards, so
/// a cut ends this cell and the loop moves to the next one.
///
/// A construction that crossed the limit cuts every variant it produced. It used to cut only the
/// first and give the later ones a fresh clock, which then measured ops built from reference math
/// that had given up part-way: a Cholesky input of `(n+1)I` instead of `M M^H + (n+1)I` factorizes
/// and checks fine, so nothing about the row would have said so.
fn run_cell(
    ops: &mut op::Op,
    run: bool,
    threads: usize,
    row: &str,
    prime_ms: u64,
    reps: usize,
    budget: Budget,
) -> Cell {
    let Budget {
        limit_s,
        deadline,
        construction_cut,
    } = budget;
    timing::arm_at(deadline);
    let cell = if construction_cut {
        Cell::cut(limit_s)
    } else if run {
        // The timed repetitions: the clock is armed inside the closure that runs them, because
        // that is the thread whose thread-local clock `timing::measure` reads.
        let (measured, cut) = measure_row(row, threads, ops, prime_ms, reps, deadline);
        match measured {
            Ok(_measured) if cut => Cell::cut(limit_s),
            Ok(measured) => Cell {
                status: "ok",
                note: ops.note.clone(),
                timed_ms: Some(measured.ms.iter().sum()),
                check_ms: Some(measured.check_ms),
                samples: measured.ms,
            },
            Err(e) => Cell::stopped_with(limit_s, cut, stopped_note(&ops.note, &e)),
        }
    } else {
        // The verification pass runs each op once on this thread: a sequential token, and the
        // clock armed above. `timed_ms` is the one call, not its preparation.
        // The limit is checked before the stage's own error is classified: a call that failed
        // after the limit passed is a cut cell, not a wrong answer.
        match ops.prepare() {
            Err(_e) if timing::expired() => Cell::cut(limit_s),
            Err(e) => Cell::stopped(limit_s, stopped_note(&ops.note, &e)),
            Ok(()) if timing::expired() => Cell::cut(limit_s),
            Ok(()) => {
                let started = Instant::now();
                let called = ops.run(op::Token::none());
                let timed_ms = started.elapsed().as_secs_f64() * 1e3;
                match called {
                    Err(_e) if timing::expired() => Cell::cut(limit_s),
                    Err(e) => Cell::stopped(limit_s, stopped_note(&ops.note, &e)),
                    Ok(()) if timing::expired() => Cell::cut(limit_s),
                    Ok(()) => {
                        let started = Instant::now();
                        let checked = ops.check();
                        let check_ms = started.elapsed().as_secs_f64() * 1e3;
                        match checked {
                            Err(_e) if timing::expired() => Cell::cut(limit_s),
                            Err(e) => Cell::stopped(limit_s, stopped_note(&ops.note, &e)),
                            Ok(()) if timing::expired() => Cell::cut(limit_s),
                            Ok(()) => Cell {
                                status: "ok",
                                note: ops.note.clone(),
                                samples: Vec::new(),
                                timed_ms: Some(timed_ms),
                                check_ms: Some(check_ms),
                            },
                        }
                    }
                }
            }
        }
    };
    timing::disarm();
    cell
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A CSV the test can read back: the rows are written as the cells finish, so the file, not a
    /// buffer, is what these tests check.
    fn csv_in(path: &std::path::Path) -> Csv {
        Csv::open(Some(&path.to_string_lossy().into_owned())).unwrap()
    }

    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("fbench-{}-{name}.csv", std::process::id()))
    }

    /// The data lines of a CSV this test just wrote.
    fn rows_of(path: &std::path::Path) -> Vec<String> {
        let text = std::fs::read_to_string(path).unwrap();
        text.lines().skip(1).map(str::to_owned).collect()
    }

    fn cli(extra: &[&str]) -> Result<Args, String> {
        let mut args: Vec<String> = [
            "fbench",
            "run",
            "--threads",
            "1",
            "--n",
            "2",
            "--csv",
            "out.csv",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect();
        args.extend(extra.iter().map(|s| (*s).to_owned()));
        parse(&args)
    }

    #[test]
    fn probe_flags_and_ordered_csv_samples() {
        assert_eq!(cli(&[]).unwrap().rotate_rows, 0);
        assert_eq!(cli(&["--rotate-rows", "2"]).unwrap().rotate_rows, 2);
        assert!(matches!(
            cli(&["--statistic", "median"]).unwrap().statistic,
            Statistic::Median
        ));
        for extra in [
            &["--rotate-rows"][..],
            &["--rotate-rows", "bad"],
            &["--reps", "0"],
            &["--statistic", "unknown"],
        ] {
            assert!(cli(extra).is_err());
        }
        assert!(parse(&["fbench", "info", "--rotate-rows", "1"].map(str::to_owned)).is_err());
        let path = scratch("probe");
        let mut csv = csv_in(&path);
        emit(
            &mut csv,
            &cli(&[]).unwrap(),
            1,
            &Dtype::F64,
            "gemm",
            Case {
                m: 2,
                n: 2,
                k: Some(2),
                batch: 1,
            },
            Order::C,
            "faer-fork",
            Cell {
                status: "ok",
                note: String::new(),
                samples: vec![3.0, 1.0, 2.0],
                timed_ms: Some(6.0),
                check_ms: Some(0.5),
            },
        );
        let row = rows_of(&path).remove(0);
        assert!(row.contains(",ok,,\"[3.0, 1.0, 2.0]\","));
        assert_eq!(summarize(&[3.0, 1.0, 2.0], Statistic::Median), Some(2.0));
        assert_eq!(
            summarize(&[4.0, 1.0, 3.0, 2.0], Statistic::Median),
            Some(2.5)
        );
        assert_eq!(summarize(&[], Statistic::Median), None);
        // The two appended columns: the timed region and the check, kept apart.
        assert!(row.ends_with(",6.000000,0.500000"), "{row}");

        // ... and `main`'s own wiring (argv -> Args -> Record) carries the profile from the
        // environment, so neither an empty `emit` nor a dropped env read stays invisible here.
        std::env::set_var("FBENCH_PROFILE", "zen2-64-smt-off");
        let args = args_from_env(&[
            "fbench".to_owned(),
            "run".to_owned(),
            "--threads".to_owned(),
            "1".to_owned(),
            "--n".to_owned(),
            "2".to_owned(),
            "--csv".to_owned(),
            "out.csv".to_owned(),
        ])
        .unwrap();
        let path = scratch("profile");
        let mut from_args = csv_in(&path);
        emit(
            &mut from_args,
            &args,
            1,
            &Dtype::F64,
            "gemm",
            Case {
                m: 2,
                n: 2,
                k: Some(2),
                batch: 1,
            },
            Order::C,
            "faer-fork",
            Cell::skipped(String::new()),
        );
        let row = rows_of(&path).remove(0);
        assert!(row.ends_with(",zen2-64-smt-off,,"), "{row}");
        std::env::remove_var("FBENCH_PROFILE");
        std::fs::remove_file(&path).ok();
    }

    /// Reaping a child that is still running kills it and returns a non-success status within the
    /// bound; it does not call a blocking `wait()`.
    #[cfg(unix)]
    #[test]
    fn reap_child_kills_a_running_child_within_the_bound() {
        let child = std::process::Command::new("sleep")
            .arg("300")
            .spawn()
            .expect("spawn sleep");
        let started = Instant::now();
        let reaped = reap_child(child);
        assert!(
            started.elapsed() < REAP_BOUND + Duration::from_secs(1),
            "reap took {:?}",
            started.elapsed()
        );
        assert!(!reaped.success(), "a killed child is not a success: {}", reaped.text());
    }

    /// An already-finished child is reaped without a signal and reports its real status.
    #[cfg(unix)]
    #[test]
    fn reap_child_reports_a_finished_child() {
        let child = std::process::Command::new("true")
            .spawn()
            .expect("spawn true");
        std::thread::sleep(Duration::from_millis(50));
        let reaped = reap_child(child);
        assert!(reaped.success(), "true should succeed: {}", reaped.text());
    }

    #[test]
    fn the_cell_limit_is_configured_with_a_default() {
        assert_eq!(cli(&["--cell-timeout", "7"]).unwrap().cell_timeout, 7);
        assert_eq!(cli(&["--cell-timeout", "0"]).unwrap().cell_timeout, 0);
        assert!(
            cli(&["--cell-timeout", "18446744073709551615"]).is_err(),
            "a limit the clock cannot represent is a configuration error, not an unlimited cell"
        );
        assert!(cli(&["--cell-timeout", "soon"]).is_err());
        if std::env::var_os(CELL_TIMEOUT_ENV).is_none() {
            assert_eq!(
                cli(&[]).unwrap().cell_timeout,
                DEFAULT_CELL_TIMEOUT_S,
                "the default limit is a starting value, not a hard-coded one"
            );
        }
    }

    /// The clock belongs to the thread that runs the cell, and a multithreaded row runs on a
    /// worker inside `spindle::with_lock`. A clock armed on the caller is invisible there, so the
    /// row would run unlimited; with the clock armed inside that closure the row is cut.
    #[test]
    fn a_multithreaded_row_reads_its_clock_on_the_worker() {
        let mut op = op::op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| {
                // Past the limit, and `Ok`: only the post-stage poll can reject it.
                std::thread::sleep(Duration::from_millis(1200));
                Ok(())
            },
            |_| Ok(()),
        );
        let cell = run_cell(&mut op, true, 2, "faer-upstream", 0, 1, Budget::new(1));
        assert_eq!(
            cell.status, "timeout",
            "a row that runs on a worker must still be cut at its limit"
        );
        assert_eq!(cell.timed_ms, None);
    }

    /// The row a cell gets when the harness cuts it off: `timeout`, naming the limit, with no
    /// partial time -- and the cell after it still runs. Without the cell clock the fake check
    /// below never returns.
    #[test]
    fn a_cut_cell_is_a_timeout_row_and_the_next_cell_still_runs() {
        let mut slow = op::op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| Ok(()),
            |_| {
                // What the serial reference math does: poll the cell clock, then give up.
                while !timing::expired() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err("residual infinite".to_owned())
            },
        );
        let cut = run_cell(&mut slow, false, 1, "faer-fork", 0, 1, Budget::new(1));
        assert_eq!(cut.status, "timeout");
        assert_eq!(cut.note, "timeout 1");
        assert_eq!(cut.timed_ms, None);
        assert_eq!(cut.check_ms, None);

        let mut fast = op::op_from((), String::new(), |_| Ok(()), |_, _tk| Ok(()), |_| Ok(()));
        let ok = run_cell(&mut fast, false, 1, "faer-fork", 0, 1, Budget::new(1));
        assert_eq!(ok.status, "ok");
        assert!(ok.timed_ms.is_some() && ok.check_ms.is_some());

        // Both rows reach the CSV: a cut cell is a row, not a hole.
        let path = scratch("cut");
        let mut csv = csv_in(&path);
        let args = cli(&[]).unwrap();
        for cell in [cut, ok] {
            emit(
                &mut csv,
                &args,
                1,
                &Dtype::F64,
                "gemm",
                Case {
                    m: 2,
                    n: 2,
                    k: Some(2),
                    batch: 1,
                },
                Order::C,
                "faer-fork",
                cell,
            );
        }
        let rows = rows_of(&path);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].contains(",timeout,timeout 1,"));
        // an unfinished region has no time, so the two appended fields are empty
        assert_eq!(rows[0].rsplit(',').take(2).collect::<Vec<_>>(), ["", ""]);
        assert!(rows[1].contains(",ok,,"));
        std::fs::remove_file(&path).ok();
    }

    /// The reference construction is the cell's work too: `cholesky` builds its input with the
    /// cubic `mul_adj_b`, and it used to run before the clock was armed, so a limit could not stop
    /// it. Building the ops must therefore already be under the cell's deadline.
    #[test]
    fn the_cell_clock_covers_the_reference_construction() {
        timing::disarm();
        let case = Case {
            m: 256,
            n: 256,
            k: None,
            batch: 1,
        };
        let deadline = timing::deadline_in(Some(Duration::from_millis(1)));
        timing::arm_at(deadline);
        let mut ops = build_ops("faer-fork", Dtype::F64, "cholesky", case, Order::C).unwrap();
        let construction_cut = timing::expired();
        assert!(
            construction_cut,
            "the reference construction must count against the cell's limit"
        );
        // Every variant built by that construction is cut, including one that would otherwise get
        // a fresh clock: the ops stand on reference math that gave up part-way.
        for op in &mut ops {
            let cell = run_cell(
                op,
                false,
                1,
                "faer-fork",
                0,
                1,
                Budget {
                    limit_s: 1,
                    deadline,
                    construction_cut,
                },
            );
            assert_eq!(
                cell.status, "timeout",
                "a cell whose construction crossed the limit must not run"
            );
        }
        timing::disarm();
    }

    /// The deadline is fixed once, when the cell (construction included) starts: a run that starts
    /// after 750 ms of construction at a 1 s limit has 250 ms, not another second. Recomputing the
    /// deadline at the start of the timed region makes this cell `ok`.
    #[test]
    fn the_cell_deadline_includes_the_reference_construction() {
        let deadline = timing::deadline_in(Some(Duration::from_secs(1)));
        timing::arm_at(deadline);
        std::thread::sleep(Duration::from_millis(750));
        let mut op = op::op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| {
                std::thread::sleep(Duration::from_millis(400));
                Ok(())
            },
            |_| Ok(()),
        );
        let cell = run_cell(
            &mut op,
            false,
            1,
            "faer-fork",
            0,
            1,
            Budget {
                limit_s: 1,
                deadline,
                construction_cut: false,
            },
        );
        assert_eq!(
            cell.status, "timeout",
            "construction time must count against the cell's limit"
        );
        timing::disarm();
    }

    /// A row that did not reach the file must not end the run as a success. The writer here is a
    /// socket whose peer is already closed, so the write really fails; without `Csv::push` setting
    /// `failed`, the run would exit 0 with a missing row.
    #[cfg(unix)]
    #[test]
    fn a_csv_that_lost_a_row_is_a_failed_run() {
        use std::os::fd::{FromRawFd, IntoRawFd};
        let (writer, reader) = std::os::unix::net::UnixStream::pair().unwrap();
        drop(reader);
        let file = unsafe { std::fs::File::from_raw_fd(writer.into_raw_fd()) };
        let mut csv = Csv {
            out: Some((std::path::PathBuf::from("a-closed-socket"), file)),
            failed: false,
        };
        emit(
            &mut csv,
            &cli(&[]).unwrap(),
            1,
            &Dtype::F64,
            "gemm",
            Case {
                m: 2,
                n: 2,
                k: Some(2),
                batch: 1,
            },
            Order::C,
            "faer-fork",
            Cell::skipped(String::new()),
        );
        assert!(
            csv.failed,
            "a write to a closed peer must be recorded as a lost row"
        );
        assert_eq!(
            format!("{:?}", exit_code(true, csv.failed)),
            format!("{:?}", ExitCode::from(2))
        );
        // A good CSV still passes, and a cell that was not `ok` is 1, not 0.
        assert_eq!(
            format!("{:?}", exit_code(true, false)),
            format!("{:?}", ExitCode::SUCCESS)
        );
        assert_eq!(
            format!("{:?}", exit_code(false, false)),
            format!("{:?}", ExitCode::from(1))
        );
    }

    /// One batch member can be two library calls: `solve`'s in-place route factors and then
    /// solves. The poll at the top of the member does not cover the second one, so the cell would
    /// start it after the limit; removing the poll between them makes this test return `Ok`.
    #[test]
    fn a_compound_batch_member_does_not_start_its_second_call_after_the_limit() {
        let mut ops = ops_for(
            "faer-fork",
            Dtype::F64,
            "solve",
            Case {
                m: 512,
                n: 512,
                k: None,
                batch: 1,
            },
            Order::C,
        )
        .unwrap();
        let mut in_place = ops.pop().unwrap();
        assert_eq!(in_place.note, "in_place");
        timing::arm(Some(Duration::from_millis(1)));
        let called = in_place.run(op::Token::none());
        assert!(
            called.is_err(),
            "the solve must not start after the factor crossed the limit"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// Every poll site in the sources is exactly the set listed in the waiting-point table, so a
    /// new wait that forgets its poll (or a deleted poll) fails here instead of in a campaign.
    #[test]
    fn waiting_points_are_enumerated_and_polled() {
        use std::collections::BTreeMap;
        let table = include_str!("../../docs/reviews/20261110-cell-timeout-waiting-points.md");
        let sources = [
            ("main.rs", include_str!("main.rs")),
            ("backend.rs", include_str!("backend.rs")),
            ("lapack.rs", include_str!("lapack.rs")),
            ("shared.rs", include_str!("shared.rs")),
            ("timing.rs", include_str!("timing.rs")),
        ];
        let is_poll = |line: &str| {
            let t = line.trim_start();
            t.starts_with("if expired()")
                || t.starts_with("if timing::expired()")
                || t.starts_with("Ok(()) if timing::expired()")
                || t.starts_with("Err(_e) if timing::expired()")
        };
        let mut listed: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
        for line in table.lines() {
            let Some(rest) = line.trim().strip_prefix("| `fbench/src/") else {
                continue;
            };
            let Some(end) = rest.find('`') else { continue };
            let Some((file, number)) = rest[..end].split_once(':') else {
                continue;
            };
            listed
                .entry(file)
                .or_default()
                .push(number.parse().expect("line number"));
        }
        for (file, source) in sources {
            let actual: Vec<usize> = source
                .lines()
                .enumerate()
                .filter(|(_, line)| is_poll(line))
                .map(|(i, _)| i + 1)
                .collect();
            assert_eq!(
                listed.get(file).cloned().unwrap_or_default(),
                actual,
                "{file}: the waiting-point table and the poll sites disagree                  (update docs/reviews/20261110-cell-timeout-waiting-points.md)"
            );
            for number in actual {
                assert!(
                    source.lines().nth(number - 1).unwrap().contains("expired()"),
                    "{file}:{number} is listed as a poll but is not one"
                );
            }
        }
    }

    /// The high-level `qr` (upstream) factorizes and then forms the thin `Q`: the poll between the
    /// two calls is what stops the second one after the limit.
    #[test]
    fn the_high_level_qr_polls_between_factor_and_thin_q() {
        let mut ops = ops_for(
            "faer-upstream",
            Dtype::F64,
            "qr",
            Case {
                m: 512,
                n: 512,
                k: None,
                batch: 1,
            },
            Order::C,
        )
        .unwrap();
        timing::arm(Some(Duration::from_millis(1)));
        let called = ops[0].run(op::Token::none());
        assert!(
            called.is_err(),
            "forming Q must not start after the factor crossed the limit"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// `householder_apply`'s check factorizes and then applies the factor: the poll between them
    /// stops the apply once the factorization has crossed the limit.
    #[test]
    fn householder_check_polls_before_applying_the_factor() {
        let mut ops = ops_for(
            "faer-fork",
            Dtype::F64,
            "householder_apply",
            Case {
                m: 512,
                n: 512,
                k: None,
                batch: 1,
            },
            Order::C,
        )
        .unwrap();
        ops[0].prepare().unwrap();
        ops[0].run(op::Token::none()).unwrap();
        timing::arm(Some(Duration::from_millis(1)));
        let checked = ops[0].check();
        assert!(
            checked.is_err(),
            "the householder apply must not start after the factorization crossed the limit"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// A limit the platform clock cannot represent is an already-expired limit, not "no limit":
    /// disarming the cell here is the failure the clock exists to prevent.
    #[test]
    fn an_unrepresentable_limit_cuts_the_cell() {
        let start = Instant::now();
        let deadline = timing::deadline_at(start, Some(Duration::from_secs(u64::MAX)));
        assert!(deadline.is_some(), "an overflowed limit must stay a limit");
        timing::arm_at(deadline);
        assert!(timing::expired(), "an overflowed limit must cut, not disarm");
        assert!(timing::cut());
        timing::disarm();
    }

    /// The compound calls that cost three review rounds: a member that factorizes and then applies,
    /// solves or forms `Q`. Each must have a poll before the second call, within the block that
    /// runs it.
    #[test]
    fn compound_library_calls_are_guarded() {
        let source = include_str!("backend.rs");
        let compounds = [
            "compute_thin_Q()",
            "solve_in_place(",
            "solve_in_place_with_conj(",
            "apply_block_householder_sequence_on_the_left_in_place_with_conj(",
            "apply_block_householder_on_the_left_in_place_with_conj(",
            "partial_piv_lu()",
            "partial_lu_of(",
            "qr_in_place(",
        ];
        for (i, line) in source.lines().enumerate() {
            if line.trim_start().starts_with("//")
                || !compounds.iter().any(|pattern| line.contains(pattern))
            {
                continue;
            }
            let start = i.saturating_sub(40);
            let guarded = source
                .lines()
                .skip(start)
                .take(i - start)
                .any(|previous| previous.contains("expired()"));
            assert!(
                guarded,
                "backend.rs:{}: no expired() poll within 40 lines of `{}`",
                i + 1,
                line.trim()
            );
        }
    }

    /// The high-level `solve` route (upstream) factors and then solves in one member: the poll
    /// between the two calls is what stops the solve after the factor crossed the limit.
    #[test]
    fn the_high_level_solve_polls_between_factor_and_solve() {
        let mut ops = ops_for(
            "faer-upstream",
            Dtype::F64,
            "solve",
            Case {
                m: 512,
                n: 512,
                k: None,
                batch: 1,
            },
            Order::C,
        )
        .unwrap();
        timing::arm(Some(Duration::from_millis(1)));
        let called = ops[0].run(op::Token::none());
        assert!(
            called.is_err(),
            "the solve must not start after the factor crossed the limit"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// The fork's owning `solve` route factors into `l`/`u` and then solves: same rule.
    #[test]
    fn the_fork_owning_solve_polls_between_factor_and_solve() {
        let mut ops = ops_for(
            "faer-fork",
            Dtype::F64,
            "solve",
            Case {
                m: 512,
                n: 512,
                k: None,
                batch: 1,
            },
            Order::C,
        )
        .unwrap();
        timing::arm(Some(Duration::from_millis(1)));
        let called = ops[0].run(op::Token::none());
        assert!(
            called.is_err(),
            "the solve must not start after the factor crossed the limit"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// The same for `qr`: the composite forms Q after it factors.
    #[test]
    fn qr_does_not_form_q_after_the_limit() {
        let mut ops = ops_for(
            "faer-fork",
            Dtype::F64,
            "qr",
            Case {
                m: 512,
                n: 512,
                k: None,
                batch: 1,
            },
            Order::C,
        )
        .unwrap();
        timing::arm(Some(Duration::from_millis(1)));
        let called = ops[0].run(op::Token::none());
        assert!(
            called.is_err(),
            "Q construction must not start after the limit"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// `householder_apply`'s `prepare` factorizes every batch member. The poll per member is what
    /// stops the second factorization after the first crossed the limit.
    #[test]
    fn a_prepare_that_factorizes_does_not_start_the_next_member_after_the_limit() {
        let mut ops = ops_for(
            "faer-fork",
            Dtype::F64,
            "householder_apply",
            Case {
                m: 512,
                n: 512,
                k: None,
                batch: 2,
            },
            Order::C,
        )
        .unwrap();
        timing::arm(Some(Duration::from_millis(1)));
        let prepared = ops[0].prepare();
        assert!(
            prepared.is_err(),
            "the second member's factorization must not start"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// A stage that returns `Err` after the limit is a cut, not a wrong answer: the limit is
    /// checked before the stage's own message is classified.
    #[test]
    fn a_stage_that_fails_after_the_limit_is_a_cut_not_a_failure() {
        let mut op = op::op_from(
            (),
            String::new(),
            |_| Ok(()),
            |_, _tk| {
                std::thread::sleep(Duration::from_millis(1300));
                Err("the library said no".to_owned())
            },
            |_| Ok(()),
        );
        let cell = run_cell(&mut op, false, 1, "faer-fork", 0, 1, Budget::new(1));
        assert_eq!(
            cell.status, "timeout",
            "an error after the limit is a cut cell, not a failure"
        );
    }

    /// A batch is many library calls under one `Op`. A limit that has passed must stop the next
    /// call, not only reject the row after every call has already run.
    #[test]
    fn a_batch_op_does_not_start_a_call_after_the_limit() {
        let mut ops = ops_for(
            "faer-upstream",
            Dtype::F64,
            "gemm",
            Case {
                m: 8,
                n: 8,
                k: Some(8),
                batch: 4,
            },
            Order::C,
        )
        .unwrap();
        timing::arm(Some(Duration::from_millis(1)));
        std::thread::sleep(Duration::from_millis(5));
        let called = ops[0].run(op::Token::none());
        assert!(
            called.is_err(),
            "the batch must poll between calls, not after all of them: {called:?}"
        );
        assert!(timing::cut());
        timing::disarm();
    }

    /// Verification calls `prepare` too, and its preparation (`householder_apply` factorizes
    /// there) can cross the limit. The call after it must not start.
    #[test]
    fn verification_does_not_run_after_a_prepare_past_the_limit() {
        let runs = Arc::new(Mutex::new(0usize));
        let r = runs.clone();
        let mut op = op::op_from(
            (),
            String::new(),
            |_| {
                // past `Budget::new(1)`'s one second
                std::thread::sleep(Duration::from_millis(1300));
                Ok(())
            },
            move |_, _tk| {
                *r.lock().unwrap() += 1;
                Ok(())
            },
            |_| Ok(()),
        );
        let cell = run_cell(&mut op, false, 1, "faer-fork", 0, 1, Budget::new(1));
        assert_eq!(cell.status, "timeout");
        assert_eq!(
            *runs.lock().unwrap(),
            0,
            "the call must not start after prepare crossed the limit"
        );
    }
}
