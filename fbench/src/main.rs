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
use std::io::Write;
use std::process::ExitCode;
use std::time::{Duration, Instant};

const USAGE: &str = "usage: fbench info [--threads N] | verify --threads N --n LIST [--batch LIST] [--dtype LIST] [--family LIST] [--regime R] [--rows LIST] [--csv PATH] | run --threads N [--cell-timeout SECS] --n LIST [--batch LIST] [--dtype LIST] [--family LIST] [--regime R] [--rows LIST] [--reps M] [--prime-ms MS] [--rotate-rows K] [--statistic best|median] --csv PATH";

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
) -> Result<Vec<f64>, String> {
    #[cfg(feature = "fork-team")]
    if threads > 1 && row == "faer-fork" {
        return faer_fork::team::with_team(threads, |par| {
            let mut token = op::Token::Fork(par);
            timing::measure(ops, prime_ms, reps, &mut token)
        });
    }
    if threads > 1 && (row == "faer-fork" || row == "faer-upstream") {
        spindle::with_lock(threads, || {
            timing::measure(ops, prime_ms, reps, &mut op::Token::none())
        })
    } else {
        timing::measure(ops, prime_ms, reps, &mut op::Token::none())
    }
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
    /// The wall-clock limit of one cell, in seconds; `0` means no limit. A cell is one `Op` and one
    /// CSV row. The limit is enforced by the parent, which kills the child that runs the cell.
    cell_timeout: u64,
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
            cell_timeout: DEFAULT_CELL_TIMEOUT_S,
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
        cell_timeout: DEFAULT_CELL_TIMEOUT_S,
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
        csv: None,
        profile: String::new(),
    };
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
            "--cell-timeout" => {
                a.cell_timeout = value()?
                    .parse()
                    .map_err(|_| "--cell-timeout is a number of seconds".to_owned())?;
            }
            "--reps" => a.reps = value()?.parse().map_err(|_| USAGE.to_owned())?,
            "--prime-ms" => a.prime_ms = value()?.parse().map_err(|_| USAGE.to_owned())?,
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

/// The per-cell limit when `--cell-timeout` is absent: an hour, which is longer than any cell the
/// campaign declares and short enough that a hung one cannot hold a run open for a day.
const DEFAULT_CELL_TIMEOUT_S: u64 = 3600;

/// How long a child that has nothing more to report gets to exit on its own before it is killed, and
/// the bound on waiting for the kill to land.
const REAP_GRACE: Duration = Duration::from_millis(250);

/// One build's work order, handed to the child as a single `;`-separated payload. The child builds
/// and measures; the parent only waits, kills and writes rows.
struct CellSpec {
    row: String,
    family: String,
    dtype: Dtype,
    order: Order,
    case: Case,
    threads: usize,
    prime_ms: u64,
    reps: usize,
    /// The first variant this child must run: a killed child is replaced by one that resumes at the
    /// variant after the one it was measured to have cut.
    from: usize,
}

impl CellSpec {
    fn encode(&self) -> String {
        format!(
            "row={};family={};dtype={};order={};m={};n={};k={};batch={};threads={};prime_ms={};reps={};from={}",
            self.row,
            self.family,
            self.dtype.label(),
            self.order.label(),
            self.case.m,
            self.case.n,
            self.case.k.map(|k| k.to_string()).unwrap_or_default(),
            self.case.batch,
            self.threads,
            self.prime_ms,
            self.reps,
            self.from
        )
    }

    fn decode(text: &str) -> Result<Self, String> {
        let field = |name: &str| -> Result<String, String> {
            text.split(';')
                .find_map(|pair| pair.strip_prefix(&format!("{name}=")))
                .map(str::to_owned)
                .ok_or_else(|| format!("--cell-spec has no {name}="))
        };
        let number = |name: &str| -> Result<usize, String> {
            field(name)?
                .parse()
                .map_err(|e| format!("--cell-spec {name}=: {e}"))
        };
        Ok(CellSpec {
            row: field("row")?,
            family: field("family")?,
            dtype: match field("dtype")?.as_str() {
                "f64" => Dtype::F64,
                "c64" => Dtype::C64,
                "f32" => Dtype::F32,
                "c32" => Dtype::C32,
                other => return Err(format!("--cell-spec dtype={other} is not a dtype")),
            },
            order: match field("order")?.as_str() {
                "c" => Order::C,
                "f" => Order::F,
                other => return Err(format!("--cell-spec order={other} is not an order")),
            },
            case: Case {
                m: number("m")?,
                n: number("n")?,
                k: match field("k")?.as_str() {
                    "" => None,
                    digits => Some(digits.parse().map_err(|e| format!("--cell-spec k=: {e}"))?),
                },
                batch: number("batch")?,
            },
            threads: number("threads")?,
            prime_ms: number("prime_ms")? as u64,
            reps: number("reps")?,
            from: number("from")?,
        })
    }
}

/// The hidden child side of `run`: one build, one `COUNT` line, then one `RESULT` per variant.
///
/// The child does not enforce the cell's limit -- it cannot interrupt a library call. It reports what
/// it measured and the parent kills it at the deadline, which is what makes the limit hold even when
/// a call never returns.
fn child_main(spec_text: &str) -> ExitCode {
    let spec = match CellSpec::decode(spec_text) {
        Ok(spec) => spec,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    // The child does the measuring, so it has to be the configuration the parent asked for: the
    // Rayon pool, the vendor's own thread count (which also turns off LAPACKE's NaN check) and the
    // faer rows' global parallelism. Without this the build ran on this process's defaults and the
    // row still said `threads=<n>`: the same cell measured on a different machine configuration.
    // A child that cannot set it up must not measure at all -- a non-zero exit makes the parent
    // write a failed row, not a `skipped` one.
    if let Err(e) = configure_threads(spec.threads) {
        eprintln!("{e}");
        return ExitCode::from(2);
    }
    // Say what it actually built, so a log (and the thread test) can check it.
    eprintln!(
        "CELL child threads={} rayon={} vendor={:?}",
        spec.threads,
        rayon::current_num_threads(),
        vendor::reported_threads()
    );
    let ops = match ops_for(&spec.row, spec.dtype, &spec.family, spec.case, spec.order) {
        Ok(ops) => ops,
        Err(note) => {
            // A family this provider cannot take is a skip, not a wrong answer.
            println!("SKIP\t{note}");
            return ExitCode::SUCCESS;
        }
    };
    // The parent needs the variant count to know which `RESULT` is the last one, and therefore which
    // row to hold until the child's exit is known.
    println!("COUNT\t{}", ops.len());
    let _ = std::io::stdout().flush();
    for (index, mut op) in ops.into_iter().enumerate() {
        if index < spec.from {
            continue;
        }
        let line = match measure_row(&spec.row, spec.threads, &mut op, spec.prime_ms, spec.reps) {
            Ok(samples) => format!(
                "RESULT\t{index}\tok\t{}\t{}",
                op.note,
                samples
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Err(e) => format!("RESULT\t{index}\tfailed\t{}; {e}\t", op.note),
        };
        println!("{line}");
        let _ = std::io::stdout().flush();
    }
    ExitCode::SUCCESS
}

/// How many fields a `RESULT` line has: `RESULT<TAB>index<TAB>status<TAB>note<TAB>samples`.
const RESULT_FIELDS: usize = 5;

/// One variant as the child reported it.
struct CellResult {
    status: &'static str,
    note: String,
    samples: Vec<f64>,
}

/// A `RESULT` line, checked. The child is the only writer of these, so a line that does not have the
/// shape below is broken output: it must never become an `ok` row, whatever it says.
fn parse_result(fields: &[&str], expected: usize, reps: usize) -> Result<CellResult, String> {
    if fields.len() != RESULT_FIELDS {
        return Err(format!(
            "RESULT has {} field(s), expected {RESULT_FIELDS}",
            fields.len()
        ));
    }
    let index: usize = fields[1]
        .parse()
        .map_err(|e| format!("RESULT index {:?}: {e}", fields[1]))?;
    if index != expected {
        return Err(format!(
            "the cell child reported variant {} when variant {} was running",
            index + 1,
            expected + 1
        ));
    }
    let status = match fields[2] {
        "ok" => "ok",
        "timeout" => "timeout",
        "failed" => "failed",
        other => return Err(format!("RESULT has an unknown status {other:?}")),
    };
    let mut samples = Vec::new();
    if status == "ok" {
        if fields[4].is_empty() {
            return Err("RESULT ok without samples".to_owned());
        }
        for sample in fields[4].split(',') {
            let value: f64 = sample
                .parse()
                .map_err(|e| format!("RESULT sample {sample:?}: {e}"))?;
            if !value.is_finite() || value < 0.0 {
                return Err(format!(
                    "RESULT sample is not a finite non-negative number: {sample}"
                ));
            }
            samples.push(value);
        }
        if samples.len() != reps {
            return Err(format!(
                "RESULT ok with {} sample(s), expected {reps}",
                samples.len()
            ));
        }
    }
    Ok(CellResult {
        status,
        note: fields[3].to_owned(),
        samples,
    })
}

enum Wake {
    Line(String),
    Timeout,
    Closed,
}

/// Wait for one line of a child's stdout, for at most `deadline`. The timeout is checked before
/// receiving, because `recv_timeout` returns an already-queued message before its own deadline.
fn wait_line(rx: &std::sync::mpsc::Receiver<String>, deadline: Option<Instant>) -> Wake {
    use std::sync::mpsc::RecvTimeoutError;
    if let Some(deadline) = deadline {
        if Instant::now() >= deadline {
            return Wake::Timeout;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        return match rx.recv_timeout(remaining) {
            Ok(line) => Wake::Line(line),
            Err(RecvTimeoutError::Timeout) => Wake::Timeout,
            Err(RecvTimeoutError::Disconnected) => Wake::Closed,
        };
    }
    match rx.recv() {
        Ok(line) => Wake::Line(line),
        Err(_) => Wake::Closed,
    }
}

/// Reap a child and report whether it ended on its own terms: `None` when it exited successfully
/// without help, `Some(reason)` when it exited non-zero or had to be killed. There is no blocking
/// `wait()`: a process that cannot be killed must not hold the run open.
fn reap(mut child: std::process::Child) -> Option<String> {
    let until = Instant::now() + REAP_GRACE;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return (!status.success()).then(|| format!("the cell child exited with {status}"))
            }
            Ok(None) => {}
            Err(e) => return Some(format!("cannot wait for the cell child: {e}")),
        }
        if Instant::now() >= until {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let killed = child.kill().is_ok();
    let _ = child.wait();
    Some(if killed {
        "the cell child was still running when its limit passed and was killed".to_owned()
    } else {
        "the cell child did not exit and could not be killed".to_owned()
    })
}

struct CellChild {
    child: std::process::Child,
    rx: std::sync::mpsc::Receiver<String>,
    reader: std::thread::JoinHandle<()>,
}

fn spawn_cell(exe: &std::path::Path, spec: &CellSpec) -> Result<CellChild, String> {
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    let mut child = Command::new(exe)
        .arg("--cell-spec")
        .arg(spec.encode())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|e| format!("cannot start the cell child: {e}"))?;
    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel::<String>();
    let reader = std::thread::spawn(move || {
        // `lines()` turns one unreadable byte into a silent end of the stream; reading whole lines and
        // decoding strictly keeps such a byte a line the protocol rejects instead.
        let mut reader = std::io::BufReader::new(stdout);
        loop {
            let mut raw = Vec::new();
            match std::io::BufRead::read_until(&mut reader, b'\n', &mut raw) {
                Ok(0) => break,
                Ok(_) => {}
                Err(_) => break,
            }
            if raw.last() == Some(&b'\n') {
                raw.pop();
                if raw.last() == Some(&b'\r') {
                    raw.pop();
                }
            }
            let line = match String::from_utf8(raw) {
                Ok(line) => line,
                Err(e) => format!("<the cell child wrote a line that is not text: {e}>"),
            };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    Ok(CellChild { child, rx, reader })
}

/// Run one build in a child process, one row per variant.
///
/// The parent does not build and does not measure. It reads `COUNT` (how many variants the build
/// produced) and one `RESULT` per variant, holds the last row until the child's exit is known, and
/// kills the child when the running variant's deadline passes. A cell the harness had to cut, and a
/// cell whose child died after reporting, are `failed`/`timeout` rows -- never `ok`.
#[allow(clippy::too_many_arguments)]
fn run_build_in_child(
    args: &Args,
    records: &mut Vec<Record>,
    all_ok: &mut bool,
    threads: usize,
    row: &str,
    dtype: &Dtype,
    family: &str,
    case: Case,
    order: Order,
) {
    let limit = (args.cell_timeout > 0).then(|| Duration::from_secs(args.cell_timeout));
    let exe = std::env::var_os("FBENCH_CELL_CHILD")
        .map(std::path::PathBuf::from)
        .map(Ok)
        .unwrap_or_else(std::env::current_exe)
        .unwrap_or_else(|_| std::path::PathBuf::from("fbench"));
    let spec_at = |from: usize| CellSpec {
        row: row.to_owned(),
        family: family.to_owned(),
        dtype: *dtype,
        order,
        case,
        threads,
        prime_ms: args.prime_ms,
        reps: args.reps,
        from,
    };
    // One row per variant, whether it was measured, cut or rejected.
    let emit_row = |records: &mut Vec<Record>,
                    all_ok: &mut bool,
                    index: usize,
                    status: &'static str,
                    note: String,
                    samples: Vec<f64>| {
        if status != "ok" && status != "skipped" {
            *all_ok = false;
        }
        let _ = index;
        emit(
            records,
            args,
            threads,
            dtype,
            family,
            case,
            order,
            row,
            (!samples.is_empty()).then_some(samples),
            status,
            note,
        );
    };

    let mut from = 0usize;
    let mut variants: Option<usize> = None;
    loop {
        let handle = match spawn_cell(&exe, &spec_at(from)) {
            Ok(handle) => handle,
            Err(e) => {
                // The started variant still gets a terminal row: a missing row is worse than a failed
                // one, because nothing then says the cell did not run.
                emit_row(records, all_ok, from, "failed", e, Vec::new());
                return;
            }
        };
        let mut expected = from;
        // The last reported variant is held until the child's exit is known: a child that reports a
        // result and then dies must not leave an `ok` row behind.
        let mut held: Option<CellResult> = None;
        let mut deadline = limit.map(|limit| Instant::now() + limit);
        let cut;

        loop {
            let wake = wait_line(&handle.rx, deadline);
            match wake {
                Wake::Line(line) => {
                    let fields: Vec<&str> = line.split('\t').collect();
                    match fields.first().copied() {
                        Some("COUNT") => {
                            if let Some(n) = fields.get(1).and_then(|n| n.parse().ok()) {
                                variants = Some(n);
                            }
                        }
                        Some("SKIP") => {
                            let note = fields.get(1).unwrap_or(&"").to_string();
                            let verdict = reap(handle.child);
                            drop(handle.reader);
                            let (status, note) = match verdict {
                                None => ("skipped", note),
                                Some(reason) => (
                                    "failed",
                                    format!(
                                        "the cell child reported a skip and then {reason}: {note}"
                                    ),
                                ),
                            };
                            emit_row(records, all_ok, from, status, note, Vec::new());
                            return;
                        }
                        Some("RESULT") => match parse_result(&fields, expected, args.reps) {
                            Ok(result) => {
                                if let Some(previous) = held.take() {
                                    let samples = previous.samples.clone();
                                    emit_row(
                                        records,
                                        all_ok,
                                        expected - 1,
                                        previous.status,
                                        previous.note,
                                        samples,
                                    );
                                }
                                held = Some(result);
                                expected += 1;
                                deadline = limit.map(|limit| Instant::now() + limit);
                            }
                            Err(e) => {
                                let verdict = reap(handle.child);
                                drop(handle.reader);
                                if let Some(previous) = held.take() {
                                    let samples = previous.samples.clone();
                                    emit_row(
                                        records,
                                        all_ok,
                                        expected - 1,
                                        previous.status,
                                        previous.note,
                                        samples,
                                    );
                                }
                                emit_row(
                                    records,
                                    all_ok,
                                    expected,
                                    "failed",
                                    match verdict {
                                        Some(reason) => format!("{e} ({reason})"),
                                        None => e,
                                    },
                                    Vec::new(),
                                );
                                return;
                            }
                        },
                        other => {
                            // A line the protocol does not define is broken output, whatever it says
                            // and wherever it appears.
                            let verdict = reap(handle.child);
                            drop(handle.reader);
                            if let Some(previous) = held.take() {
                                let samples = previous.samples.clone();
                                emit_row(
                                    records,
                                    all_ok,
                                    expected - 1,
                                    previous.status,
                                    previous.note,
                                    samples,
                                );
                            }
                            emit_row(
                                records,
                                all_ok,
                                expected,
                                "failed",
                                format!(
                                    "the cell child wrote a line that is not part of the protocol: {:?}",
                                    other.unwrap_or(&line)
                                ),
                                Vec::new(),
                            );
                            let _ = verdict;
                            return;
                        }
                    }
                }
                Wake::Timeout => {
                    let verdict = reap(handle.child);
                    drop(handle.reader);
                    match (variants, held.take()) {
                        // The result that was held is the last variant's: the child hung after
                        // reporting it, so it is settled by the kill, not recorded as a measurement.
                        (Some(n), Some(_previous)) if expected >= n => {
                            emit_row(
                                records,
                                all_ok,
                                n - 1,
                                "failed",
                                format!(
                                    "the final cell did not complete: {}",
                                    verdict
                                        .unwrap_or_else(|| "the cell child was cut off".to_owned())
                                ),
                                Vec::new(),
                            );
                            return;
                        }
                        // The deadline passed while a later variant was running: the held row measured
                        // what it measured, and this variant is the cut one.
                        (Some(n), previous) if expected < n => {
                            if let Some(previous) = previous {
                                let samples = previous.samples.clone();
                                emit_row(
                                    records,
                                    all_ok,
                                    expected - 1,
                                    previous.status,
                                    previous.note,
                                    samples,
                                );
                            }
                            emit_row(
                                records,
                                all_ok,
                                expected,
                                "timeout",
                                format!("timeout {}", args.cell_timeout),
                                Vec::new(),
                            );
                            cut = true;
                            from = expected + 1;
                        }
                        // The child died before it said how many variants the build has: nothing more
                        // is known, so the variant that was started is the cut one.
                        _ => {
                            emit_row(
                                records,
                                all_ok,
                                from,
                                "timeout",
                                format!("timeout {}", args.cell_timeout),
                                Vec::new(),
                            );
                            return;
                        }
                    }
                    if cut {
                        break;
                    }
                    return;
                }
                Wake::Closed => {
                    let verdict = reap(handle.child);
                    drop(handle.reader);
                    match held.take() {
                        // The held row is the last variant's, and the child's exit is now known.
                        Some(previous) => {
                            let (status, note) = match verdict {
                                None => (previous.status, previous.note),
                                Some(reason) => (
                                    "failed",
                                    format!("the final cell did not complete: {reason}"),
                                ),
                            };
                            let samples = previous.samples.clone();
                            emit_row(
                                records,
                                all_ok,
                                expected - 1,
                                status,
                                note,
                                if status == "ok" { samples } else { Vec::new() },
                            );
                        }
                        // Nothing was reported: the child died before its first result.
                        None => emit_row(
                            records,
                            all_ok,
                            expected,
                            "failed",
                            format!(
                                "the cell child stopped before reporting a result: {}",
                                verdict.unwrap_or_else(|| "it exited successfully".to_owned())
                            ),
                            Vec::new(),
                        ),
                    }
                    return;
                }
            }
        }
        if !cut {
            return;
        }
    }
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    // Hidden child mode: one build, measured in its own process so the parent can kill it at the
    // cell's deadline. It is not part of the recorder's CLI.
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
    let mut records = Vec::new();
    let mut all_ok = true;

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
                                        &mut records,
                                        &args,
                                        threads,
                                        dtype,
                                        family,
                                        case_for_note(*shape, family, *batch),
                                        order,
                                        row,
                                        None,
                                        "skipped",
                                        note.clone(),
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
                            if run {
                                // One build in its own process: the parent writes the rows and
                                // enforces the cell's limit by killing it.
                                run_build_in_child(
                                    &args,
                                    &mut records,
                                    &mut all_ok,
                                    threads,
                                    row,
                                    dtype,
                                    family,
                                    case,
                                    order,
                                );
                                continue;
                            }
                            let ops = match ops_for(row, *dtype, family, case, order) {
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
                                        &mut records,
                                        &args,
                                        threads,
                                        dtype,
                                        family,
                                        case,
                                        order,
                                        row,
                                        None,
                                        "skipped",
                                        note,
                                    );
                                    continue;
                                }
                            };
                            for mut ops in ops {
                                let label = format!(
                                    "{row} {} {} {} {}",
                                    family,
                                    dtype.label(),
                                    order.label(),
                                    case.shape()
                                );
                                let outcome = ops
                                    .prepare()
                                    .and_then(|()| {
                                        // the verification pass runs each op once, outside any
                                        // budget: a sequential token
                                        ops.run(op::Token::none())
                                    })
                                    .and_then(|()| ops.check());
                                match outcome {
                                    Ok(()) => {
                                        println!("ok {label} {}", ops.note);
                                        emit(
                                            &mut records,
                                            &args,
                                            threads,
                                            dtype,
                                            family,
                                            case,
                                            order,
                                            row,
                                            None,
                                            "ok",
                                            ops.note.clone(),
                                        );
                                    }
                                    Err(e) => {
                                        let e = format!("{}; {e}", ops.note);
                                        println!("MISMATCH {label} {e}");
                                        all_ok = false;
                                        emit(
                                            &mut records,
                                            &args,
                                            threads,
                                            dtype,
                                            family,
                                            case,
                                            order,
                                            row,
                                            None,
                                            "failed",
                                            e,
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if !run {
        if all_ok {
            println!("all comparisons within tolerance");
        }
        if args.csv.is_none() {
            return if all_ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            };
        }
    }

    let path = args
        .csv
        .as_ref()
        .expect("run or verification census needs --csv");
    let mut text = String::from(
        "regime,family,dtype,order,m,n,k,batch,row,threads,total_ms,per_item_us,status,note,samples_ms,rayon_threads,vendor_threads,profile\n",
    );
    for record in &records {
        text.push_str(&record.line());
        text.push('\n');
    }
    if let Err(e) = std::fs::write(path, text) {
        eprintln!("cannot write {path}: {e}");
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

#[allow(clippy::too_many_arguments)]
fn emit(
    records: &mut Vec<Record>,
    args: &Args,
    threads: usize,
    dtype: &Dtype,
    family: &str,
    case: Case,
    order: Order,
    row: &str,
    samples: Option<Vec<f64>>,
    status: &'static str,
    note: String,
) {
    records.push(Record {
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
        total_ms: samples.as_ref().and_then(|s| summarize(s, args.statistic)),
        samples_ms: samples.unwrap_or_default(),
        status,
        note,
        profile: args.profile.clone(),
    });
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut records = Vec::new();
        emit(
            &mut records,
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
            Some(vec![3.0, 1.0, 2.0]),
            "ok",
            String::new(),
        );
        assert_eq!(records[0].total_ms, Some(1.0));
        assert!(records[0].line().contains(",ok,,\"[3.0, 1.0, 2.0]\","));
        assert_eq!(summarize(&[3.0, 1.0, 2.0], Statistic::Median), Some(2.0));
        assert_eq!(
            summarize(&[4.0, 1.0, 3.0, 2.0], Statistic::Median),
            Some(2.5)
        );
        assert_eq!(summarize(&[], Statistic::Median), None);
        assert_eq!(records[0].samples_ms, [3.0, 1.0, 2.0]);
        records[0].rayon_threads = 17;
        records[0].vendor_threads = Some(19);
        assert!(records[0].line().ends_with(",17,19,"));
        // The profile column is the recorder's declaration, passed down through the
        // environment; without one the column is empty, never a guess.
        records[0].profile = "zen2-64-smt-off".to_owned();
        assert!(records[0].line().ends_with(",17,19,zen2-64-smt-off"));

        // ... and `main`'s own wiring (argv -> Args -> Record) carries the profile from the
        // environment, so neither an empty `emit` nor a dropped env read stays invisible here.
        std::env::set_var("FBENCH_PROFILE", "zen2-64-smt-off");
        let args = args_from_env(&["fbench".to_owned(), "run".to_owned(), "--threads".to_owned(),
                                   "1".to_owned(), "--n".to_owned(), "2".to_owned(),
                                   "--csv".to_owned(), "out.csv".to_owned()])
            .unwrap();
        let mut from_args = Vec::new();
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
            None,
            "skipped",
            String::new(),
        );
        assert_eq!(from_args[0].profile, "zen2-64-smt-off");
        assert!(from_args[0].line().ends_with(",zen2-64-smt-off"));
        std::env::remove_var("FBENCH_PROFILE");
    }
}
