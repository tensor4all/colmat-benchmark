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

/// A cell that outlives its 1 s limit is killed, its row is `timeout`, its `CELL done` pairs with
/// the `CELL start`, the kill names the pid, that pid is gone, and the next build still runs.
#[test]
fn a_killed_cell_leaves_a_row_and_the_next_cell_runs() {
    let (stderr, csv, code) = run("kill", "512", "svd_thin", &["--cell-timeout", "1"]);
    assert_eq!(code, 1, "a cut cell is a non-zero run: {stderr}");

    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 4, "two orders x two svd_thin variants: {csv}");
    for row in &rows {
        assert!(
            row.contains(",timeout,timeout 1,"),
            "a killed cell must be a timeout row: {row}"
        );
    }

    let starts = count_lines(&stderr, "CELL start");
    let dones = count_lines(&stderr, "CELL done");
    assert_eq!(
        starts, dones,
        "every start needs a done, including the killed cell:\n{stderr}"
    );
    for line in stderr.lines().filter(|l| l.starts_with("CELL done")) {
        assert!(
            line.contains("variant=") && line.contains("status=timeout") && line.contains("cut=timeout"),
            "a killed cell closes with its variant and cut=timeout: {line}"
        );
    }

    let pids: Vec<&str> = stderr
        .lines()
        .filter(|line| line.starts_with("CELL kill"))
        .filter_map(|line| {
            line.split_whitespace()
                .find_map(|field| field.strip_prefix("pid="))
        })
        .collect();
    assert!(
        pids.len() >= 2,
        "each cut variant is killed and respawned, so at least one kill per order: {stderr}"
    );
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

    // The second order ran after the first was killed. A cut variant is respawned (`from=`), so
    // there can be more than one build line per order.
    assert!(
        count_lines(&stderr, "CELL build") >= 2,
        "the second order must run: {stderr}"
    );
    assert!(
        stderr.lines().any(|l| l.starts_with("CELL start") && l.contains("id=2")),
        "the run continued past the kill: {stderr}"
    );
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
        8,
        "the event table changed: update docs/reviews/20261110-cell-timeout-invariant.md and this test"
    );

    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fake_child.sh");
    for (event, at, expected) in cases {
        assert!(
            matches!(event.as_str(), "fast" | "cut" | "error" | "skip"),
            "the event table has an event this test does not implement: {event}"
        );
        let name = format!("table-{event}-{at}");
        let csv = csv_path(&name);
        let output = Command::new(env!("CARGO_BIN_EXE_fbench"))
            .args([
                "run", "--threads", "1", "--n", "8", "--family", "gemm", "--dtype", "f64",
                "--rows", "faer-fork", "--reps", "1", "--prime-ms", "0", "--cell-timeout", "1",
                "--csv",
            ])
            .arg(&csv)
            .env("FBENCH_CELL_CHILD", script)
            .env("FBENCH_TEST_VARIANTS", "3")
            .env("FAKE_MODE", &event)
            .env("FAKE_AT", at.to_string())
            .env("FAKE_N", "3")
            .output()
            .expect("run fbench with the fake child");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let body = std::fs::read_to_string(&csv).unwrap_or_default();
        std::fs::remove_file(&csv).ok();

        let mut starts: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        let mut dones: BTreeMap<String, Vec<(usize, String)>> = BTreeMap::new();
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

        // Two orders, one build each.
        assert_eq!(starts.len(), 2, "{name}: {stderr}");
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
