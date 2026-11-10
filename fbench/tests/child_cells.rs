//! The parent/child contract of a cell, end to end: the cell's limit is enforced by killing the
//! child, broken output is never an `ok` row, and a cell that dies after its last result fails the
//! run. These run the real binary (`CARGO_BIN_EXE_fbench`), so the hidden `--cell-spec` path, the
//! kill and the settlement are what is tested.

#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

fn csv_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fbench-child-{}-{name}.csv", std::process::id()))
}

/// One `fbench run` over two orders and one family. `fake` selects the fake child and its variant
/// count; without it the real child runs the real build.
struct Run {
    code: Option<i32>,
    csv: Vec<Vec<String>>,
    /// The harness's stderr; the child inherits it, so a diagnostic the child prints is here.
    stderr: String,
    /// Pids still in the harness's process group the moment it exited, sampled before this test's
    /// own cleanup: a child the harness failed to kill is visible only until then.
    survivors: Vec<u32>,
}

fn run(name: &str, fake: Option<(&str, usize)>, extra: &[&str]) -> Run {
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;
    let csv = csv_path(name);
    let err = std::env::temp_dir().join(format!("fbench-{}-{name}.err", std::process::id()));
    let stderr = std::fs::File::create(&err).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_fbench"));
    command
        .args([
            "run", "--threads", "1", "--n", "8", "--family", "gemm", "--dtype", "f64", "--rows",
            "faer-fork", "--reps", "1", "--prime-ms", "0", "--cell-timeout", "1", "--csv",
        ])
        .arg(&csv)
        .args(extra)
        .stdout(Stdio::null())
        .stderr(Stdio::from(stderr))
        .process_group(0);
    if let Some((mode, variants)) = fake {
        command
            .env("FBENCH_CELL_CHILD", concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_child.sh"))
            .env("FAKE_MODE", mode)
            .env("FAKE_N", variants.to_string());
    }
    let mut child = command.spawn().expect("spawn fbench");
    let group = child.id();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let survivors = loop {
        if let Some(_status) = child.try_wait().expect("try_wait") {
            let alive = group_members(group);
            // Cleanup: the harness is the writer of its rows, so anything it left behind is the
            // orphan this test is looking for.
            let _ = Command::new("kill")
                .args(["-KILL", &format!("-{group}")])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
            break alive;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "{name}: fbench did not finish within 30 s"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let code = child.wait().expect("wait fbench").code();
    let text = std::fs::read_to_string(&csv).unwrap_or_default();
    let stderr = std::fs::read_to_string(&err).unwrap_or_default();
    std::fs::remove_file(&csv).ok();
    std::fs::remove_file(&err).ok();
    Run {
        code,
        csv: records(&text),
        stderr,
        survivors,
    }
}

/// Pids in one process group. `ps` that cannot answer must fail the test, not report an empty group:
/// "nothing is alive" is the answer these tests are looking for.
fn group_members(group: u32) -> Vec<u32> {
    let out = Command::new("ps")
        .args(["-A", "-o", "pid=,pgid="])
        .output()
        .expect("ps -A -o pid=,pgid=");
    assert!(out.status.success(), "ps failed: {out:?}");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid: u32 = fields.next()?.parse().ok()?;
            let pgid: u32 = fields.next()?.parse().ok()?;
            (pgid == group && pid != group).then_some(pid)
        })
        .collect()
}

/// CSV records, read the way Python's `csv` module reads by default: a bare CR or LF ends a record
/// only outside a quoted field.
fn records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' => quoted = true,
            ',' => record.push(std::mem::take(&mut field)),
            '\r' | '\n' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                record.push(std::mem::take(&mut field));
                if !record.is_empty() {
                    records.push(std::mem::take(&mut record));
                }
            }
            other => field.push(other),
        }
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

fn statuses(run: &Run) -> Vec<String> {
    run.csv
        .iter()
        .skip(1)
        .map(|row| row.get(12).cloned().unwrap_or_default())
        .collect()
}

/// A run under its limit measures every variant and writes one `ok` row per order.
#[test]
fn a_finished_run_writes_one_ok_row_per_variant() {
    let run = run("ok", None, &["--cell-timeout", "60"]);
    assert_eq!(run.code, Some(0), "{:?}", run.csv);
    assert_eq!(statuses(&run), ["ok", "ok"], "{:?}", run.csv);
    assert!(
        run.survivors.is_empty(),
        "the harness left processes behind: {:?}",
        run.survivors
    );
}

/// The child measures the configuration the parent asked for. It used to run the build without ever
/// calling `configure_threads`, so it built the pool it liked (every core) while the row said
/// `threads=1`: the same cell, a different machine configuration. The child now configures itself --
/// pool, vendor thread count, the faer rows' parallelism -- and re-checks the result before it
/// builds anything, and it says what it ended up with.
///
/// The registry arm this repository builds in CI links no vendor library, so `--threads 2` is
/// refused outright (by the parent too, before this change); a vendor arm would also assert
/// `vendor=Some(2)` here.
#[test]
fn the_child_measures_with_the_threads_it_was_given() {
    let run = run("threads", None, &["--cell-timeout", "60"]);
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    assert_eq!(statuses(&run), ["ok", "ok"], "{:?}", run.csv);
    assert!(
        run.stderr.contains("CELL child threads=1 rayon=1 vendor=None"),
        "the child must configure --threads and check it before it builds: {}",
        run.stderr
    );
}

/// The limit is enforced from outside: a child that never reports is killed, its cell is a `timeout`
/// row, the run fails, and nothing of the child is left.
#[test]
fn a_cell_past_its_limit_is_killed_and_leaves_no_child() {
    let run = run("kill", Some(("hang", 1)), &[]);
    assert_eq!(run.code, Some(1), "{:?}", run.csv);
    assert_eq!(statuses(&run), ["timeout", "timeout"], "{:?}", run.csv);
    assert!(
        run.survivors.is_empty(),
        "the harness killed the cell but left {:?} running",
        run.survivors
    );
}

/// Broken output is never an `ok` row: a line with a field too many, and one whose sample is not a
/// number, each close their cell as `failed` and fail the run.
#[test]
fn broken_output_is_never_an_ok_row() {
    for (name, mode) in [("bad-result", "bad_result"), ("bad-numbers", "bad_numbers")] {
        let run = run(name, Some((mode, 1)), &[]);
        assert_eq!(run.code, Some(1), "{mode}: {:?}", run.csv);
        assert_eq!(statuses(&run), ["failed", "failed"], "{mode}: {:?}", run.csv);
    }
}

/// The last cell's row is held until the child's exit is known: a child that reports every result and
/// then dies leaves a `failed` row, not an `ok` one.
#[test]
fn a_final_child_that_dies_after_its_last_result_fails_the_run() {
    let run = run("crash-after-last", Some(("crash_after_last", 1)), &[]);
    assert_eq!(run.code, Some(1), "{:?}", run.csv);
    assert_eq!(statuses(&run), ["failed", "failed"], "{:?}", run.csv);
    let note = run.csv.get(1).and_then(|row| row.get(13)).cloned().unwrap_or_default();
    assert!(note.contains("did not complete"), "{note}");
}

/// The same when the final child hangs instead of exiting: it is killed, its row is `failed`, and the
/// run fails -- a killed cell is not a measurement.
#[test]
fn a_final_child_that_hangs_is_killed_and_the_run_fails() {
    let run = run("final-hang", Some(("hang_after", 1)), &[]);
    assert_eq!(run.code, Some(1), "{:?}", run.csv);
    assert_eq!(statuses(&run), ["failed", "failed"], "{:?}", run.csv);
    assert!(
        run.survivors.is_empty(),
        "the harness left {:?} running",
        run.survivors
    );
}
