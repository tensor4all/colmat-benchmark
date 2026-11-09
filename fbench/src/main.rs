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
use std::process::ExitCode;
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

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
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
                            heartbeat(
                                "start",
                                row,
                                family,
                                dtype,
                                order,
                                case,
                                format!(" threads={threads} reps={}", args.reps),
                            );
                            // The cell starts here: one deadline covers the reference
                            // construction and the timed repetitions, and it is never recomputed.
                            // If construction crossed it, every op it produced stands on reference
                            // math that gave up part-way, so none of them is measured.
                            let deadline = timing::deadline_in(limit);
                            timing::arm_at(deadline);
                            let built = build_ops(row, *dtype, family, case, order);
                            let construction_cut = timing::expired();
                            let ops = match built {
                                Ok(ops) => ops,
                                Err(note) => {
                                    if !run {
                                        println!(
                                            "skip {row} {} {} {} {} {note}",
                                            family,
                                            dtype.label(),
                                            order.label(),
                                            case.shape()
                                        );
                                    }
                                    emit(
                                        &mut csv,
                                        &args,
                                        threads,
                                        dtype,
                                        family,
                                        case,
                                        order,
                                        row,
                                        Cell::skipped(note),
                                    );
                                    continue;
                                }
                            };
                            for mut ops in ops {
                                let cell = run_cell(
                                    &mut ops,
                                    run,
                                    threads,
                                    row,
                                    args.prime_ms,
                                    args.reps,
                                    Budget {
                                        limit_s: args.cell_timeout,
                                        deadline,
                                        construction_cut,
                                    },
                                );
                                if !run {
                                    let label = format!(
                                        "{row} {} {} {} {}",
                                        family,
                                        dtype.label(),
                                        order.label(),
                                        case.shape()
                                    );
                                    // The recorder reads `MISMATCH` from stdout. A cell the harness
                                    // cut off says so in its own word: the CSV status is what a
                                    // script reads, and the log is what a human reads at 3 a.m.
                                    match cell.status {
                                        "ok" => println!("ok {label} {}", cell.note),
                                        "timeout" => println!("TIMEOUT {label} {}", cell.note),
                                        _ => println!("MISMATCH {label} {}", cell.note),
                                    }
                                }
                                heartbeat(
                                    "done",
                                    row,
                                    family,
                                    dtype,
                                    order,
                                    case,
                                    format!(
                                        " {} total_ms={}",
                                        cell.status,
                                        milliseconds(summarize(&cell.samples, args.statistic))
                                    ),
                                );
                                if cell.status != "ok" && cell.status != "skipped" {
                                    all_ok = false;
                                }
                                emit(
                                    &mut csv,
                                    &args,
                                    threads,
                                    dtype,
                                    family,
                                    case,
                                    order,
                                    row,
                                    cell,
                                );
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

    #[test]
    fn the_cell_limit_is_configured_with_a_default() {
        assert_eq!(cli(&["--cell-timeout", "7"]).unwrap().cell_timeout, 7);
        assert_eq!(cli(&["--cell-timeout", "0"]).unwrap().cell_timeout, 0);
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
        assert_eq!(
            rows[0].rsplit(',').take(2).collect::<Vec<_>>(),
            ["", ""]
        );
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
