//! The parent/child contract of a cell: the child can be killed, the killed cell still leaves a
//! `timeout` row and a matching `CELL done`, and the next cell runs normally.
//!
//! These run the real binary (`CARGO_BIN_EXE_fbench`) so the hidden `--cell-spec` path, the kill,
//! and the CSV/CSV-adjacent heartbeat are exercised end to end.

#![cfg(unix)]

use std::path::PathBuf;
use std::process::Command;

fn csv_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("fbench-child-{}-{name}.csv", std::process::id()))
}

fn run(name: &str, n: &str, family: &str, extra: &[&str]) -> (String, String, i32) {
    let csv = csv_path(name);
    let mut command = Command::new(env!("CARGO_BIN_EXE_fbench"));
    command
        .args([
            "run",
            "--threads",
            "1",
            "--n",
            n,
            "--family",
            family,
            "--dtype",
            "f64",
            "--rows",
            "faer-fork",
            "--reps",
            "1",
            "--prime-ms",
            "0",
            "--csv",
        ])
        .arg(&csv)
        .args(extra);
    let output = command.output().expect("run fbench");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let csv_text = std::fs::read_to_string(&csv).unwrap_or_default();
    std::fs::remove_file(&csv).ok();
    (stderr, csv_text, output.status.code().unwrap_or(-1))
}

fn count_lines(text: &str, prefix: &str) -> usize {
    text.lines().filter(|line| line.starts_with(prefix)).count()
}

/// Run the parent with a fake child under a wall deadline. Every wait inside the parent is
/// bounded; this is the outer bound for the test itself, so a regression that removes the parent's
/// deadline fails instead of hanging in `sleep 300`.
fn run_fake_with_deadline(
    name: &str,
    mode: &str,
    at: usize,
    n: usize,
    seconds: u64,
) -> (Option<i32>, String, String) {
    use std::os::unix::process::CommandExt;
    use std::process::Stdio;
    let csv = csv_path(name);
    let err_path = std::env::temp_dir().join(format!("fbench-{}-{name}.err", std::process::id()));
    let err_file = std::fs::File::create(&err_path).unwrap();
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_child.sh");
    let mut command = Command::new(env!("CARGO_BIN_EXE_fbench"));
    command
        .args([
            "run", "--threads", "1", "--n", "8", "--family", "gemm", "--dtype", "f64",
            "--rows", "faer-fork", "--reps", "1", "--prime-ms", "0", "--cell-timeout", "1",
            "--csv",
        ])
        .arg(&csv)
        .env("FBENCH_CELL_CHILD", script)
        .env("FBENCH_TEST_VARIANTS", n.to_string())
        .env("FAKE_MODE", mode)
        .env("FAKE_AT", at.to_string())
        .env("FAKE_N", n.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::from(err_file))
        .process_group(0);
    let mut child = command.spawn().expect("spawn fbench");
    let pid = child.id();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();
            let csv_text = std::fs::read_to_string(&csv).unwrap_or_default();
            std::fs::remove_file(&csv).ok();
            std::fs::remove_file(&err_path).ok();
            return (status.code(), stderr, csv_text);
        }
        if std::time::Instant::now() >= deadline {
            // The parent is in its own process group; kill the group so the fake child cannot
            // outlive the test.
            let _ = Command::new("kill")
                .args(["-KILL", &format!("-{pid}")])
                .status();
            let _ = child.kill();
            let _ = child.wait();
            let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();
            let csv_text = std::fs::read_to_string(&csv).unwrap_or_default();
            std::fs::remove_file(&csv).ok();
            std::fs::remove_file(&err_path).ok();
            return (None, stderr, csv_text);
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn run_fake(name: &str, mode: &str, at: usize, n: usize) -> (String, String, i32) {
    let (code, stderr, csv) = run_fake_with_deadline(name, mode, at, n, 10);
    (
        stderr,
        csv,
        code.expect("the fake-child run must finish within the test's 10 s deadline"),
    )
}

/// The first variant of a build is hard-killed by the parent, deterministically: the fake child
/// sleeps before reporting anything, so this never depends on a real cell taking longer than the
/// limit on a particular host. The remaining variants are respawned and run.
#[test]
fn a_first_variant_hard_kill_leaves_a_row_and_the_next_variant_runs() {
    let (stderr, csv, code) = run_fake("hard-kill", "hang", 0, 3);
    assert_eq!(code, 1, "a cut variant is a non-zero run: {stderr}");

    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 6, "two orders x three variants: {csv}");
    for (build, chunk) in rows.chunks(3).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["timeout", "ok", "ok"],
            "build {build} after a first-variant kill: {csv}"
        );
    }

    let starts: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("CELL start"))
        .collect();
    let dones: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("CELL done"))
        .collect();
    assert_eq!(starts.len(), 6, "one start per variant: {stderr}");
    assert_eq!(dones.len(), 6, "one done per variant: {stderr}");

    let pids: Vec<&str> = stderr
        .lines()
        .filter(|line| line.starts_with("CELL kill"))
        .filter_map(|line| {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix("pid="))
        })
        .collect();
    assert_eq!(pids.len(), 2, "one hard kill per order: {stderr}");
    for pid in pids {
        let alive = Command::new("ps")
            .args(["-p", pid, "-o", "pid="])
            .output()
            .expect("ps");
        assert!(
            !alive.status.success(),
            "killed child {pid} is still in ps:\n{}",
            String::from_utf8_lossy(&alive.stdout)
        );
    }
}

/// A run that finishes under its limit pairs every start with a done and writes `ok` rows.
#[test]
fn a_finished_run_pairs_every_start_with_a_done() {
    let (stderr, csv, code) = run("ok", "64", "gemm", &["--cell-timeout", "60"]);
    assert_eq!(code, 0, "{stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one gemm cell per layout: {csv}");
    for row in &rows {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.get(12).copied(), Some("ok"), "{row}");
    }
    assert_eq!(
        count_lines(&stderr, "CELL start"),
        count_lines(&stderr, "CELL done"),
        "{stderr}"
    );
    let starts: Vec<&str> = stderr
        .lines()
        .filter(|l| l.starts_with("CELL start"))
        .collect();
    assert!(
        starts.iter().all(|l| l.contains("variant=")),
        "each start is identifiable: {stderr}"
    );
}

/// The terminal-row table: one `CELL start`, one terminal row and one `CELL done` per variant, for
/// every event and position. The table lives in `docs/reviews/20261110-cell-timeout-invariant.md`;
/// the row-count assert makes a new event (or a removed one) fail until the table is updated.
#[test]
fn every_cell_event_leaves_exactly_one_terminal_row() {
    use std::collections::BTreeMap;

    let table = include_str!("../../docs/reviews/20261110-cell-timeout-invariant.md");
    let mut cases: Vec<(String, usize, Vec<String>)> = Vec::new();
    for line in table.lines() {
        let cells: Vec<&str> = line.trim().trim_matches('|').split('|').map(str::trim).collect();
        if cells.len() != 3 {
            continue;
        }
        let event = cells[0];
        let at = match cells[1] {
            "first" => 0,
            "middle" => 1,
            "last" => 2,
            "all" => 0,
            _ => continue,
        };
        let statuses: Vec<String> = cells[2].split(',').map(|s| s.trim().to_owned()).collect();
        if statuses.is_empty() {
            continue;
        }
        cases.push((event.to_owned(), at, statuses));
    }
    assert_eq!(
        cases.len(),
        9,
        "the event table changed: update docs/reviews/20261110-cell-timeout-invariant.md and this test"
    );

    for (event, at, expected) in cases {
        assert!(
            matches!(event.as_str(), "fast" | "cut" | "hang" | "error" | "skip"),
            "the event table has an event this test does not implement: {event}"
        );
        let name = format!("table-{event}-{at}");
        let (code, stderr, body) = run_fake_with_deadline(&name, &event, at, 3, 10);
        assert!(
            code.is_some(),
            "{name}: the parent did not finish within the test's 10 s deadline: {stderr}"
        );

        let mut starts: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut dones: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
        let mut kills: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for line in stderr.lines() {
            let id = line
                .split_whitespace()
                .find_map(|f| f.strip_prefix("id="))
                .unwrap_or("?")
                .to_owned();
            if let Some(variant) = line.split_whitespace().find_map(|f| f.strip_prefix("variant=")) {
                let k: usize = variant
                    .split('/')
                    .next()
                    .unwrap()
                    .parse()
                    .unwrap_or(0);
                if line.starts_with("CELL start") {
                    starts.entry(id.clone()).or_default().push(k);
                } else if line.starts_with("CELL kill") {
                    if let Some(pid) = line
                        .split_whitespace()
                        .find_map(|f| f.strip_prefix("pid="))
                    {
                        kills.entry(id.clone()).or_default().push(pid.to_owned());
                    }
                } else if line.starts_with("CELL done") {
                    let status = line
                        .split_whitespace()
                        .find_map(|f| f.strip_prefix("status="))
                        .unwrap_or("?")
                        .to_owned();
                    dones.entry(id.clone()).or_default().push((k, status));
                }
            }
        }

        // Two orders, one build each. The ids must match: a build whose done lines all went
        // missing must fail here even though the remaining id still has a complete start/done set.
        assert_eq!(starts.len(), 2, "{name}: {stderr}");
        let start_ids: Vec<&String> = starts.keys().collect();
        let done_ids: Vec<&String> = dones.keys().collect();
        assert_eq!(
            start_ids, done_ids,
            "{name}: every start id needs a done id: starts {start_ids:?} dones {done_ids:?}\n{stderr}"
        );
        for (id, variants) in &starts {
            let per_build = if event == "skip" { 1 } else { 3 };
            assert_eq!(
                variants.len(),
                per_build,
                "{name} id={id}: one start per variant: {stderr}"
            );
            let mut sorted = variants.clone();
            sorted.sort_unstable();
            let want: Vec<usize> = (1..=per_build).collect();
            assert_eq!(sorted, want, "{name} id={id}: starts are {variants:?}");
        }
        for (id, variants) in &dones {
            let per_build = if event == "skip" { 1 } else { 3 };
            assert_eq!(
                variants.len(),
                per_build,
                "{name} id={id}: one done per variant: {stderr}"
            );
            let mut sorted: Vec<usize> = variants.iter().map(|(k, _)| *k).collect();
            sorted.sort_unstable();
            let want: Vec<usize> = (1..=per_build).collect();
            assert_eq!(sorted, want, "{name} id={id}: dones are {variants:?}");
            // A done with the wrong status (or a done for the wrong variant) is not an invariant.
            for (k, status) in variants {
                assert_eq!(
                    status,
                    &expected[k - 1],
                    "{name} id={id}: variant {k} done status: {stderr}"
                );
            }
        }

        if event == "hang" {
            for (id, pids) in &kills {
                assert_eq!(
                    pids.len(),
                    1,
                    "{name} id={id}: the first variant is cut by one SIGKILL: {stderr}"
                );
                let alive = Command::new("ps")
                    .args(["-p", &pids[0], "-o", "pid="])
                    .output()
                    .expect("ps");
                assert!(
                    !alive.status.success(),
                    "{name} id={id}: killed child {} is still in ps",
                    pids[0]
                );
            }
            assert_eq!(kills.len(), 2, "{name}: one hard kill per build: {stderr}");
        }

        let rows: Vec<&str> = body.lines().skip(1).collect();
        let per_build = if event == "skip" { 1 } else { expected.len() };
        assert_eq!(rows.len(), per_build * 2, "{name}: {body}");
        for (build, chunk) in rows.chunks(per_build).enumerate() {
            let got: Vec<String> = chunk
                .iter()
                .map(|row| row.split(',').nth(12).unwrap_or("?").to_owned())
                .collect();
            assert_eq!(got, expected, "{name}: build {build} statuses: {body}");
        }
    }
}

/// The child trusts the parent pid it was given, not a `getppid()` taken after it started: if the
/// parent died in between, `getppid()` returns 1 and the child would treat the orphan state as
/// normal. A spec naming a pid that is not the parent must exit before doing any work.
#[test]
fn the_child_exits_when_the_given_parent_is_not_its_parent() {
    let spec = "id=1;from=0;variants=1;parent_pid=1;run=1;row=faer-fork;family=gemm;dtype=f64;order=c;m=8;column=8;k=8;batch=1;threads=1;prime_ms=0;reps=1;limit_s=0";
    let output = Command::new(env!("CARGO_BIN_EXE_fbench"))
        .args(["--cell-spec", spec])
        .output()
        .expect("run the child");
    assert_eq!(
        output.status.code(),
        Some(137),
        "the child must exit when its given parent is not its parent: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn a_child_that_hangs_after_its_last_result_does_not_stall_the_run() {
    let (code, stderr, _csv) = run_fake_with_deadline("hang-after", "hang_after", 0, 3, 10);
    assert_eq!(
        code,
        Some(0),
        "the parent must finish after the last result: {stderr}"
    );
    assert_eq!(count_lines(&stderr, "CELL start"), 6, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 6, "{stderr}");
}

/// An unsupported family is a `skip` in `verify`, not a wrong answer: the reporter reads
/// `MISMATCH` from stdout, so a skipped route must not print it.
#[test]
fn verify_reports_an_unsupported_family_as_skip_not_mismatch() {
    let output = Command::new(env!("CARGO_BIN_EXE_fbench"))
        .args([
            "verify", "--threads", "1", "--family", "eig", "--dtype", "f64", "--rows",
            "faer-fork", "--n", "8",
        ])
        .output()
        .expect("run fbench verify");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        stdout.contains("skip faer-fork eig"),
        "the unsupported family must be a skip line: {stdout}"
    );
    assert!(
        !stdout.contains("MISMATCH"),
        "a skip must not print MISMATCH: {stdout}"
    );
    assert_eq!(output.status.code(), Some(0), "a skip is not a failure");
}
