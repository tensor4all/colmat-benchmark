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
    assert_eq!(pids.len(), 2, "one kill per faer-fork order: {stderr}");
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

    // The second build (the F-order cells) ran after the first was killed.
    assert_eq!(count_lines(&stderr, "CELL build"), 2, "{stderr}");
    assert!(
        stderr
            .lines()
            .filter(|l| l.starts_with("CELL start"))
            .count()
            >= 4,
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
