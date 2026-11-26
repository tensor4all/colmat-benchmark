//! What the harness itself observed about a run, written beside its CSV: the CPU set the process
//! was actually allowed to run on (and the core class of each CPU), and a marker when the run did
//! not complete. A row's label (`zen5-cpu`, `threads=4`) is a claim; these files are the evidence a
//! reader checks it against (colmat-benchmark#65, #130).

/// Parse a kernel CPU list (`4-7,16-23`) into sorted CPU numbers. `None` for anything malformed.
pub fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
                if a > b {
                    return None;
                }
                cpus.extend(a..=b);
            }
            None => cpus.push(part.trim().parse().ok()?),
        }
    }
    cpus.sort_unstable();
    cpus.dedup();
    Some(cpus)
}

/// The inverse of [`parse_cpu_list`]: `[4,5,6,7,16]` -> `4-7,16`.
pub fn format_cpu_list(cpus: &[usize]) -> String {
    let mut out: Vec<String> = Vec::new();
    let mut i = 0;
    while i < cpus.len() {
        let mut j = i;
        while j + 1 < cpus.len() && cpus[j + 1] == cpus[j] + 1 {
            j += 1;
        }
        out.push(if j > i { format!("{}-{}", cpus[i], cpus[j]) } else { cpus[i].to_string() });
        i = j + 1;
    }
    out.join(",")
}

/// The CPUs this process may run on, from `Cpus_allowed_list` of a `/proc/<pid>/status` text.
pub fn allowed_cpus(status: &str) -> Option<Vec<usize>> {
    let line = status.lines().find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?;
    parse_cpu_list(line)
}

/// Group CPUs by their reported performance index: `124:4-11;196:0-3`. A CPU whose index is
/// unknown is grouped under `unknown`, never guessed.
pub fn class_summary(perf: &[(usize, Option<u32>)]) -> String {
    let mut groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
    for (cpu, p) in perf {
        let key = p.map_or("unknown".to_string(), |p| p.to_string());
        groups.entry(key).or_default().push(*cpu);
    }
    groups
        .into_iter()
        .map(|(k, v)| format!("{k}:{}", format_cpu_list(&v)))
        .collect::<Vec<_>>()
        .join(";")
}

fn highest_perf(cpu: usize) -> Option<u32> {
    std::fs::read_to_string(format!(
        "/sys/devices/system/cpu/cpu{cpu}/cpufreq/amd_pstate_highest_perf"
    ))
    .ok()?
    .trim()
    .parse()
    .ok()
}

/// The observed affinity of this process as `key: value` lines, or `None` off Linux.
pub fn affinity_report(threads: usize) -> Option<String> {
    let cpus = allowed_cpus(&std::fs::read_to_string("/proc/self/status").ok()?)?;
    let perf: Vec<_> = cpus.iter().map(|&c| (c, highest_perf(c))).collect();
    Some(format!(
        "observed_cpus: {}\ncpu_count: {}\nthreads: {threads}\namd_pstate_highest_perf: {}\nsource: Cpus_allowed_list of /proc/self/status in the harness process\n",
        format_cpu_list(&cpus),
        cpus.len(),
        class_summary(&perf),
    ))
}

/// The marker body for a run that did not complete: `statuses` are the status and note of every
/// row. `None` when every row is acceptable (`ok` or `skipped`).
pub fn partial_marker(rows: &[(&str, &str)]) -> Option<String> {
    let bad: Vec<_> = rows.iter().filter(|(s, _)| !matches!(*s, "ok" | "skipped")).collect();
    if bad.is_empty() {
        return None;
    }
    let mut counts: std::collections::BTreeMap<&str, usize> = Default::default();
    for (s, _) in &bad {
        *counts.entry(s).or_default() += 1;
    }
    let counts: Vec<_> = counts.iter().map(|(s, n)| format!("{s}={n}")).collect();
    let mut out = format!(
        "partial: true\nrows: {}\nnot_ok: {}\nstatuses: {}\n",
        rows.len(),
        bad.len(),
        counts.join(",")
    );
    for (s, note) in bad.iter().take(5) {
        out.push_str(&format!("first_failure: {s}: {}\n", note.replace('\n', " ")));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_lists_round_trip() {
        assert_eq!(parse_cpu_list("4-7,16-17,20"), Some(vec![4, 5, 6, 7, 16, 17, 20]));
        assert_eq!(format_cpu_list(&[4, 5, 6, 7, 16, 17, 20]), "4-7,16-17,20");
        assert_eq!(parse_cpu_list("7-4"), None);
        assert_eq!(parse_cpu_list("x"), None);
    }

    #[test]
    fn allowed_cpus_reads_the_status_line() {
        let s = "Name:\tfbench\nCpus_allowed:\tff0\nCpus_allowed_list:\t4-11\n";
        assert_eq!(allowed_cpus(s), Some((4..=11).collect()));
        assert_eq!(allowed_cpus("Name:\tx\n"), None);
    }

    #[test]
    fn classes_group_by_perf_and_never_guess() {
        let perf = [(3, Some(196)), (4, Some(124)), (5, Some(124)), (6, None)];
        assert_eq!(class_summary(&perf), "124:4-5;196:3;unknown:6");
    }

    #[test]
    fn marker_only_for_a_run_with_a_non_ok_row() {
        assert_eq!(partial_marker(&[("ok", ""), ("skipped", "n")]), None);
        let m = partial_marker(&[("ok", ""), ("failed", "signal: 9 (SIGKILL)"), ("timeout", "t")]).unwrap();
        assert!(m.contains("partial: true") && m.contains("failed=1,timeout=1"), "{m}");
        assert!(m.contains("signal: 9 (SIGKILL)"), "{m}");
    }
}
