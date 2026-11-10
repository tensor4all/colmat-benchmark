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

/// Parse CSV the way Python's `csv` module does by default: a bare CR or LF ends a record unless it
/// is inside a quoted field. This is what a consumer of the harness's CSV sees, so a field the
/// harness failed to quote shows up here as extra records -- not as a `lines()` count that still
/// looks right.
fn csv_records(text: &str) -> Vec<Vec<String>> {
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

/// Run the parent with a fake child under a wall deadline. Every wait inside the parent is
/// bounded; this is the outer bound for the test itself, so a regression that removes the parent's
/// deadline fails instead of hanging in `sleep 300`.
/// Pids still in `group` (a process group id) right now. The test's kill below is cleanup; a test
/// that asserts "the harness killed and reaped its child" has to look *before* that cleanup, or it
/// passes whether or not the harness did anything.
fn group_survivors(group: u32) -> Vec<u32> {
    let out = Command::new("ps")
        .args(["-A", "-o", "pid=,pgid="])
        .output()
        .expect("ps -A -o pid=,pgid=");
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

fn run_fake_with_options(
    name: &str,
    mode: &str,
    at: usize,
    n: usize,
    reps: usize,
    seconds: u64,
    extra: &[&str],
) -> (Option<i32>, String, String, Vec<u32>) {
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
            "--rows", "faer-fork", "--reps", &reps.to_string(), "--prime-ms", "0",
            "--cell-timeout", "1", "--csv",
        ])
        .arg(&csv)
        .env("FBENCH_CELL_CHILD", script)
        .env("FBENCH_TEST_VARIANTS", n.to_string())
        .env("FAKE_MODE", mode)
        .env("FAKE_AT", at.to_string())
        .env("FAKE_N", n.to_string())
        .args(extra)
        .stdout(Stdio::null())
        .stderr(Stdio::from(err_file))
        .process_group(0);
    let mut child = command.spawn().expect("spawn fbench");
    let pid = child.id();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(seconds);
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            // The parent is gone. Snapshot what is still in its process group *first*: anything the
            // harness failed to kill or reap is visible only until the cleanup below.
            let survivors = group_survivors(pid);
            let _ = Command::new("kill")
                .args(["-KILL", &format!("-{pid}")])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();
            let csv_text = std::fs::read_to_string(&csv).unwrap_or_default();
            std::fs::remove_file(&csv).ok();
            std::fs::remove_file(&err_path).ok();
            return (status.code(), stderr, csv_text, survivors);
        }
        if std::time::Instant::now() >= deadline {
            // The parent is in its own process group; kill the group so the fake child cannot
            // outlive the test.
            let _ = Command::new("kill")
                .args(["-KILL", &format!("-{pid}")])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status();
            let _ = child.kill();
            let _ = child.wait();
            let stderr = std::fs::read_to_string(&err_path).unwrap_or_default();
            let csv_text = std::fs::read_to_string(&csv).unwrap_or_default();
            std::fs::remove_file(&csv).ok();
            std::fs::remove_file(&err_path).ok();
            return (None, stderr, csv_text, Vec::new());
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn run_fake_with_deadline(
    name: &str,
    mode: &str,
    at: usize,
    n: usize,
    seconds: u64,
) -> (Option<i32>, String, String) {
    let (code, stderr, csv, _survivors) = run_fake_with_options(name, mode, at, n, 1, seconds, &[]);
    (code, stderr, csv)
}

/// The same run, but also reporting which pids were still alive in the harness's process group the
/// moment it exited -- the evidence a "the harness killed its child" assertion needs.
fn run_fake_with_survivors(
    name: &str,
    mode: &str,
    at: usize,
    n: usize,
    seconds: u64,
) -> (Option<i32>, String, String, Vec<u32>) {
    run_fake_with_options(name, mode, at, n, 1, seconds, &[])
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
    // The survivors are sampled the moment the harness exits, before this test's own cleanup: a
    // post-hoc `ps -p` would pass even if the harness had killed and reaped nothing.
    let (code, stderr, csv, survivors) = run_fake_with_survivors("hard-kill", "hang", 0, 3, 10);
    let code = code.expect("the fake-child run must finish within the test's 10 s deadline");
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
    for pid in &pids {
        let pid: u32 = pid.parse().expect("a pid");
        assert!(
            !survivors.contains(&pid),
            "the harness killed child {pid} but did not reap it: still alive {survivors:?}\n{stderr}"
        );
    }
    assert!(
        survivors.is_empty(),
        "the harness must leave nothing running in its process group: {survivors:?}\n{stderr}"
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
        9,
        "the event table changed: update docs/reviews/20261110-cell-timeout-invariant.md and this test"
    );

    for (event, at, expected) in cases {
        assert!(
            matches!(event.as_str(), "fast" | "cut" | "hang" | "error" | "skip"),
            "the event table has an event this test does not implement: {event}"
        );
        let name = format!("table-{event}-{at}");
        let (code, stderr, body, survivors) = run_fake_with_survivors(&name, &event, at, 3, 10);
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
                let pid: u32 = pids[0].parse().expect("a pid");
                assert!(
                    !survivors.contains(&pid),
                    "{name} id={id}: the harness killed child {pid} but left it alive {survivors:?}"
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

/// A child that hangs after its last result still lets the parent finish (bounded reap), and the
/// kill it takes then fails the run: a final cell the parent had to kill is not a measurement.
#[test]
fn a_child_that_hangs_after_its_last_result_does_not_stall_the_run() {
    let (code, stderr, csv) = run_fake_with_deadline("hang-after", "hang_after", 0, 3, 10);
    assert_eq!(
        code,
        Some(1),
        "the parent must finish after the last result, and the kill must fail the run: {stderr}"
    );
    assert_eq!(count_lines(&stderr, "CELL start"), 6, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 6, "{stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 6, "two orders x three variants: {csv}");
    for (build, chunk) in rows.chunks(3).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["ok", "ok", "failed"],
            "build {build}: only the child the parent had to kill is failed: {csv}"
        );
    }
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

/// A clean exit says nothing about whether the child's output has finished: a child that reports its
/// last result, forks a grandchild that keeps stdout open and exits 0 leaves the stream unfinished.
/// Settling that as `ok` accepts output the harness never read.
#[test]
fn an_unfinished_output_stream_after_the_last_result_is_a_failed_row() {
    let (code, stderr, csv, survivors) = run_fake_with_survivors("fork-after-last", "fork_after_last", 0, 1, 10);
    assert_eq!(
        code,
        Some(1),
        "an output stream still open after the last result is a failed run: {stderr}"
    );
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.get(12).copied(), Some("failed"), "{row}");
        let note = fields.get(13).copied().unwrap_or("");
        assert!(note.contains("output"), "the note says why: {row}");
    }
    // The harness leaves its own child reaped; the forked grandchild is the test's cleanup, which
    // is exactly why the snapshot is taken first.
    assert!(
        !survivors.is_empty(),
        "the forked grandchild is expected to still be running: {survivors:?}"
    );
    assert_eq!(count_lines(&stderr, "CELL done"), 2, "{stderr}");
}

/// The final variant's child reports its last result and then never exits. The parent kills it, and
/// that kill has to reach the row and the run: acceptance ③ (a final cell that dies must fail the
/// run), which `Reaped::Killed` used to keep as a successful measurement.
#[test]
fn a_final_child_that_had_to_be_killed_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("final-hang", "hang_after", 0, 1);
    assert_eq!(code, 1, "the final kill is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
        let note = row.split(',').nth(13).unwrap_or("");
        assert!(
            note.contains("did not exit") || note.contains("did not complete"),
            "the note says why: {row}"
        );
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 2, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 2, "{stderr}");
    assert!(stderr.contains("final-reap"), "{stderr}");
}

/// A child that cannot be started closes the already-started variant with a `failed` row and a
/// matching done, instead of exiting with a header-only CSV.
#[test]
fn a_child_that_cannot_start_closes_the_variant_with_a_failed_row() {
    let csv = csv_path("spawn-failure");
    let output = Command::new(env!("CARGO_BIN_EXE_fbench"))
        .args([
            "run", "--threads", "1", "--n", "8", "--family", "gemm", "--dtype", "f64",
            "--rows", "faer-fork", "--reps", "1", "--prime-ms", "0", "--cell-timeout", "1",
            "--csv",
        ])
        .arg(&csv)
        .env("FBENCH_CELL_CHILD", "/nonexistent/fbench-child")
        .output()
        .expect("run fbench");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let body = std::fs::read_to_string(&csv).unwrap_or_default();
    std::fs::remove_file(&csv).ok();

    assert_eq!(
        output.status.code(),
        Some(1),
        "a failed cell is a non-zero run, not a crash: {stderr}"
    );
    let rows: Vec<&str> = body.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {body}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    let starts = count_lines(&stderr, "CELL start");
    let dones = count_lines(&stderr, "CELL done");
    assert_eq!(starts, 2, "one start per order: {stderr}");
    assert_eq!(dones, 2, "one done per order: {stderr}");
    assert!(
        stderr.contains("spawn-failed"),
        "the spawn failure is logged: {stderr}"
    );
}

/// A child that crashes after reporting a result is not the cooperative construction cut. Only
/// "a timeout result, then a successful exit" gets the respawn-without-a-row exception; a non-zero
/// exit after an `ok` result closes the pending variant as failed.
#[test]
fn a_crash_after_a_result_is_not_a_cooperative_exit() {
    let (stderr, csv, code) = run_fake("crash-after-result", "error_once", 1, 3);
    assert_eq!(code, 1, "the crash is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 6, "two orders x three variants: {csv}");
    for (build, chunk) in rows.chunks(3).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["ok", "failed", "ok"],
            "build {build}: the crash must close variant 2 as failed: {csv}"
        );
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 6, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 6, "{stderr}");
}

/// A malformed `RESULT` (or an index that is not the running variant) closes that variant as
/// failed, writes its done, and reaps the child; it does not return early with no row.
#[test]
fn a_malformed_result_closes_the_variant_and_reaps_the_child() {
    let (stderr, csv, code) = run_fake("bad-result", "bad_result", 0, 2);
    assert_eq!(code, 1, "a protocol error is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 4, "two orders x two variants: {csv}");
    for (build, chunk) in rows.chunks(2).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["failed", "ok"],
            "build {build}: the malformed result is a failed cell: {csv}"
        );
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 4, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 4, "{stderr}");
    assert!(
        stderr.contains("protocol-error"),
        "the protocol error is logged: {stderr}"
    );
}

/// An index a `usize` can hold but this build cannot is a failed row, not a panic: the parent used
/// to abort while formatting "variant 18446744073709551615 while 1 was running", which left no row
/// and no reason. The index is checked where it is parsed, once.
#[test]
fn an_index_beyond_the_variants_is_a_failed_row_not_a_panic() {
    let (stderr, csv, code) = run_fake("huge-index", "huge_index", 0, 2);
    assert_eq!(
        code, 1,
        "a protocol error is a failed run, not a crash: {stderr}"
    );
    assert!(
        !stderr.contains("panicked"),
        "the parent must not panic on a broken index: {stderr}"
    );
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 4, "two orders x two variants: {csv}");
    for (build, chunk) in rows.chunks(2).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["failed", "ok"],
            "build {build}: the out-of-range index is a failed cell: {csv}"
        );
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 4, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 4, "{stderr}");
    assert!(stderr.contains("out of range"), "{stderr}");
}

/// A child that reports its last result and then exits non-zero must not leave an `ok` run.
#[test]
fn a_crash_after_the_last_result_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("crash-after-last", "crash_after_last", 0, 1);
    assert_eq!(code, 1, "the post-result crash is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 2, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 2, "{stderr}");
}

/// The child is read to the end after its last result, so extra output there is not silently
/// accepted: the last row is only `ok` if the child said nothing more.
#[test]
fn output_after_the_last_result_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("garbage-after-last", "garbage_after_last", 0, 3);
    assert_eq!(
        code, 1,
        "extra output after the last result is a failed run: {stderr}"
    );
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 6, "two orders x three variants: {csv}");
    for (build, chunk) in rows.chunks(3).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["ok", "ok", "failed"],
            "build {build}: the extra line after the last result fails it: {csv}"
        );
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 6, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 6, "{stderr}");
    assert!(stderr.contains("extra-output"), "{stderr}");
}

/// The parent also reads what the child says while the child is exiting: a line that lands after
/// the drain bound but before the exit is still the child's output, not a clean row.
#[test]
fn late_output_during_the_reap_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("late-garbage", "late_garbage", 0, 2);
    assert_eq!(
        code, 1,
        "extra output during the reap is a failed run: {stderr}"
    );
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 4, "two orders x two variants: {csv}");
    for (build, chunk) in rows.chunks(2).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["ok", "failed"],
            "build {build}: the late line fails the last row: {csv}"
        );
    }
    assert!(stderr.contains("extra-output"), "{stderr}");
}

/// An unreadable byte is a line the protocol rejects, not the end of the child's output: decoding
/// the stream with `lines()` turned it into a clean EOF and the last row stayed `ok`.
#[test]
fn invalid_utf8_after_the_last_result_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("bad-utf8", "bad_utf8_after_last", 0, 2);
    assert_eq!(code, 1, "an undecodable tail is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 4, "two orders x two variants: {csv}");
    for (build, chunk) in rows.chunks(2).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["ok", "failed"],
            "build {build}: the undecodable line fails the last row: {csv}"
        );
    }
    assert!(stderr.contains("extra-output"), "{stderr}");
}

/// A line the protocol does not define is broken output wherever it appears. It used to be dropped
/// while a result was still pending, so whether the same line was accepted depended on its position:
/// `garbage` before a valid `RESULT` was ignored, the same line after the last result was not.
#[test]
fn an_unknown_line_before_a_result_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("garbage-before", "garbage_before_result", 0, 3);
    assert_eq!(code, 1, "an unknown line is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 6, "two orders x three variants: {csv}");
    for (build, chunk) in rows.chunks(3).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["failed", "ok", "ok"],
            "build {build}: the unknown line fails variant 1, the rest are respawned: {csv}"
        );
    }
    // The failed variant is closed once, and the respawned child still covers variants 2 and 3.
    assert_eq!(count_lines(&stderr, "CELL start"), 6, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 6, "{stderr}");
    assert!(stderr.contains("protocol-error"), "{stderr}");
}

/// An `ok` row is only a measurement if the numbers it *derives* are numbers too. Every field of the
/// result is finite, but `per_item_us` scales it by 1000 and the median averages two samples, so
/// finite samples can still overflow to `inf` -- which then reached the CSV on an `ok` row.
#[test]
fn a_sample_that_overflows_the_derived_measurement_is_a_failed_row() {
    // `best` keeps the sample itself and overflows at the scale; `median` overflows in the summary.
    for (name, mode, reps, extra, refused) in [
        ("huge-sample", "huge_sample", 1, &[][..], "per_item_us"),
        (
            "huge-sample-median",
            "huge_sample_pair",
            2,
            &["--statistic", "median"][..],
            "total_ms",
        ),
    ] {
        let (code, stderr, csv, _survivors) =
            run_fake_with_options(name, mode, 0, 1, reps, 10, extra);
        assert_eq!(code, Some(1), "{mode}: a non-finite measurement is a failed run: {stderr}");
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 2, "{mode}: one row per order: {csv}");
        for row in &rows {
            let status = row.split(',').nth(12).unwrap_or("?");
            assert_eq!(status, "failed", "{mode}: {row}");
        }
        // The note may name the value it rejected; the numeric columns may not carry one.
        for row in &rows {
            let fields: Vec<&str> = row.split(',').collect();
            for (index, name) in [
                (10, "total_ms"),
                (11, "per_item_us"),
                (14, "samples_ms"),
                (18, "timed_ms"),
                (19, "check_ms"),
            ] {
                let value = fields.get(index).copied().unwrap_or("");
                assert!(
                    !value.contains("inf") && !value.contains("NaN"),
                    "{mode}: {name} is not a number: {row}"
                );
            }
        }
        // The row says which derived number it refused.
        for row in &rows {
            let note = row.split(',').nth(13).unwrap_or("");
            assert!(note.contains("not finite"), "{mode}: {row}");
            assert!(note.contains(refused), "{mode}: the note should name {refused}: {row}");
        }
        assert_eq!(count_lines(&stderr, "CELL done"), 2, "{mode}: {stderr}");
    }
}

/// A note is forwarded into the CSV. The writer quoted LF, `"` and `,` but not a bare CR, so a note
/// containing one split the row into two records for any RFC 4180 reader (Python's `csv`, and the
/// campaign's `summarize_samples.py`).
#[test]
fn a_note_with_a_carriage_return_does_not_break_the_csv() {
    let (stderr, csv, code) = run_fake("cr-note", "cr_note", 0, 1);
    assert_eq!(code, 0, "a well-formed line with a CR in its note is still a run: {stderr}");
    let records = csv_records(&csv);
    assert_eq!(
        records.len(),
        3,
        "a header and one row per order, not four records from two rows: {records:?}"
    );
    for row in &records[1..] {
        assert_eq!(row.len(), 20, "20 columns per row: {row:?}");
        assert_eq!(row[12], "ok", "{row:?}");
        assert_eq!(row[13], "foo\rbar", "the note survives verbatim: {row:?}");
    }
}

/// `COUNT` is the first line the child writes, and it is a protocol line like the others: two fields,
/// the second an integer. It used to be accepted whatever it said, so a broken `COUNT` was broken
/// output recorded as success -- the same fail-open as the other line kinds had.
#[test]
fn a_count_with_the_wrong_shape_is_a_failed_row() {
    for (name, mode) in [("bad-count", "bad_count"), ("bad-count-payload", "bad_count_payload")] {
        let (stderr, csv, code) = run_fake(name, mode, 0, 1);
        assert_eq!(code, 1, "{mode}: a malformed COUNT is a failed run: {stderr}");
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 2, "{mode}: one row per order: {csv}");
        for row in &rows {
            let status = row.split(',').nth(12).unwrap_or("?");
            assert_eq!(status, "failed", "{mode}: {row}");
        }
        assert!(stderr.contains("protocol-error"), "{mode}: {stderr}");
        assert_eq!(count_lines(&stderr, "CELL done"), 2, "{mode}: {stderr}");
    }
}

/// A `SKIP` has a fixed shape too: two fields. A stray field used to be dropped and the line became a
/// `skipped` row, which is broken output recorded as success on the other settlement path.
#[test]
fn a_skip_with_the_wrong_field_count_is_a_failed_row() {
    for (name, mode) in [("skip-extra-field", "skip_extra_field"), ("skip-short", "skip_short")] {
        let (stderr, csv, code) = run_fake(name, mode, 0, 1);
        assert_eq!(code, 1, "{mode}: a malformed SKIP is a failed run: {stderr}");
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 2, "{mode}: one row per order: {csv}");
        for row in &rows {
            let status = row.split(',').nth(12).unwrap_or("?");
            assert_eq!(status, "failed", "{mode}: {row}");
        }
        assert!(stderr.contains("field(s)"), "{mode}: {stderr}");
        assert_eq!(count_lines(&stderr, "CELL done"), 2, "{mode}: {stderr}");
    }
}

/// A `RESULT` line has a fixed shape: seven tab-separated fields. A stray field is broken output
/// (and it would otherwise carry `garbage` past both the number checks and the extra-output check,
/// because it belongs to the same line), and so is a missing one.
#[test]
fn a_result_with_the_wrong_field_count_is_a_failed_row() {
    for (name, mode) in [("extra-field", "extra_field"), ("short-result", "short_result")] {
        let (stderr, csv, code) = run_fake(name, mode, 0, 1);
        assert_eq!(code, 1, "{mode}: a malformed RESULT is a failed run: {stderr}");
        let rows: Vec<&str> = csv.lines().skip(1).collect();
        assert_eq!(rows.len(), 2, "{mode}: one row per order: {csv}");
        for row in &rows {
            let status = row.split(',').nth(12).unwrap_or("?");
            assert_eq!(status, "failed", "{mode}: {row}");
        }
        assert!(stderr.contains("field(s)"), "{mode}: {stderr}");
    }
}

/// A `SKIP` is not a licence to say anything else: output after it is broken output, so the row is
/// `failed`, not `skipped`.
#[test]
fn output_after_a_skip_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("garbage-after-skip", "garbage_after_skip", 0, 1);
    assert_eq!(code, 1, "output after a skip is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 2, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 2, "{stderr}");
    assert!(stderr.contains("protocol-error"), "{stderr}");
}

/// Undecodable bytes before the last result are broken output too, not a line to skip: decoding
/// them leniently would let a child that printed garbage then a valid `RESULT` pass.
#[test]
fn invalid_utf8_before_a_result_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("bad-utf8-before", "bad_utf8_before_result", 0, 1);
    assert_eq!(code, 1, "an undecodable line is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert!(stderr.contains("protocol-error"), "{stderr}");
}

/// The reader keeps `lines()`' own rule for a line ending -- one `\n`, then at most one `\r` -- so
/// a malformed sample is not repaired into a number by trimming every trailing carriage return.
#[test]
fn an_extra_carriage_return_is_not_trimmed_into_a_valid_sample() {
    let (stderr, csv, code) = run_fake("extra-cr", "extra_cr_sample", 0, 1);
    assert_eq!(code, 1, "a stray carriage return is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert!(stderr.contains("protocol-error"), "{stderr}");
}

/// A last line without a newline keeps its carriage return, as `lines()` did: stripping it here
/// repaired the sample into a number.
#[test]
fn a_bare_carriage_return_at_eof_is_not_trimmed_into_a_valid_sample() {
    let (stderr, csv, code) = run_fake("bare-cr", "bare_cr_at_eof", 0, 1);
    assert_eq!(code, 1, "a stray carriage return is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert!(stderr.contains("protocol-error"), "{stderr}");
}

/// Garbage/NaN timings and samples are not an accepted measurement.
#[test]
fn garbage_numbers_are_not_an_ok_row() {
    let (stderr, csv, code) = run_fake("bad-numbers", "bad_numbers", 0, 1);
    assert_eq!(code, 1, "a bad numeric result is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order, not a header-only CSV: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert!(stderr.contains("protocol-error"), "{stderr}");
}

/// An `ok` result with fewer samples than repetitions is not accepted.
#[test]
fn a_short_sample_count_is_not_an_ok_row() {
    let (code, stderr, csv, _survivors) =
        run_fake_with_options("short-samples", "short_samples", 0, 1, 3, 10, &[]);
    assert_eq!(code, Some(1), "a short sample set is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order, not a header-only CSV: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert!(stderr.contains("expected 3"), "{stderr}");
}

/// A skip is only a skip if the child exits successfully. SKIP followed by exit 3 is a failed cell,
/// not a successful skipped row.
#[test]
fn a_skip_followed_by_a_crash_is_a_failed_row() {
    let (stderr, csv, code) = run_fake("skip-exit3", "skip_exit3", 0, 1);
    assert_eq!(code, 1, "a skip that crashes is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one row per order: {csv}");
    for row in &rows {
        let status = row.split(',').nth(12).unwrap_or("?");
        assert_eq!(status, "failed", "{row}");
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 2, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 2, "{stderr}");
    assert!(stderr.contains("skip-failed"), "{stderr}");
}

/// A child that floods COUNT lines without a result must still be killed at the variant deadline:
/// `recv_timeout` returns queued messages before its deadline, so the clock is checked first.
#[test]
fn a_count_flood_does_not_escape_the_deadline() {
    let (stderr, csv, code) = run_fake("count-flood", "count_flood", 0, 1);
    assert_eq!(code, 1, "the deadline cuts the flooding child: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 2, "one cut row per order: {csv}");
    for row in &rows {
        assert!(row.contains(",timeout,"), "{row}");
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 2, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 2, "{stderr}");
    assert!(
        count_lines(&stderr, "CELL kill") >= 2,
        "the flooding child is killed: {stderr}"
    );
}

/// A SKIP that arrives after a result is a protocol error for the pending variant; it must not
/// finish variant 1 a second time or drop the remaining variants.
#[test]
fn a_late_skip_is_a_failed_cell_and_the_rest_run() {
    let (stderr, csv, code) = run_fake("late-skip", "late_skip", 0, 3);
    assert_eq!(code, 1, "a late SKIP is a failed run: {stderr}");
    let rows: Vec<&str> = csv.lines().skip(1).collect();
    assert_eq!(rows.len(), 6, "two orders x three variants: {csv}");
    for (build, chunk) in rows.chunks(3).enumerate() {
        let statuses: Vec<&str> = chunk
            .iter()
            .map(|row| row.split(',').nth(12).unwrap_or("?"))
            .collect();
        assert_eq!(
            statuses,
            ["ok", "failed", "ok"],
            "build {build}: the late SKIP fails variant 2 and variant 3 runs: {csv}"
        );
    }
    for id in ["id=1", "id=2"] {
        let starts: Vec<&str> = stderr
            .lines()
            .filter(|l| l.starts_with("CELL start") && l.contains(id))
            .collect();
        assert_eq!(starts.len(), 3, "{id}: one start per variant: {stderr}");
        assert!(
            starts.iter().any(|l| l.contains("variant=3/3")),
            "{id}: variant 3 must start: {stderr}"
        );
    }
    assert_eq!(count_lines(&stderr, "CELL start"), 6, "{stderr}");
    assert_eq!(count_lines(&stderr, "CELL done"), 6, "{stderr}");
}
