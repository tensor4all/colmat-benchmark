"""The patch-resolution preflight must keep reporting and recovering, not just refuse.

`build_for_faer_rev.sh` refuses to build when a `[patch]` would be ignored, which is correct — but an
earlier version discarded cargo's own error, so a checkout whose dependency set was newer than the
local cargo cache looked like a resolution bug. The preflight now retries once online after printing
cargo's message. Nothing in CI runs that script, so this test pins both branches with a stub `cargo`,
including a mutation of the retry path (replacing it with `exit 1`, the old behaviour) that must fail.
"""
from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO / "scripts" / "faer_patch_preflight.sh"

STUB = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$STUB_LOG"
if [[ "$*" == *"--offline"* && "${STUB_OFFLINE:-fail}" == "fail" ]]; then
    echo "error: failed to download t4a-gemm v0.20.1" >&2
    exit 101
fi
if [[ "$*" != *"--offline"* && "${STUB_ONLINE:-ok}" == "fail" ]]; then
    echo "error: no route to crates.io" >&2
    exit 101
fi
cat "$STUB_METADATA"
"""


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        cargo = self.bin / "cargo"
        cargo.write_text(STUB)
        cargo.chmod(cargo.stat().st_mode | stat.S_IEXEC)
        self.log = self.root / "calls.log"
        self.fork = self.root / "fork" / "faer" / "Cargo.toml"
        self.upstream = self.root / "upstream" / "faer" / "Cargo.toml"
        for path in (self.fork, self.upstream):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("[package]\n")
        self.metadata = self.root / "metadata.json"
        self.write_metadata(fork=self.fork, upstream=self.upstream)

    def tearDown(self):
        self.directory.cleanup()

    def write_metadata(self, *, fork, upstream):
        self.metadata.write_text(json.dumps({"packages": [
            {"name": "t4a-faer", "manifest_path": str(fork)},
            {"name": "faer", "manifest_path": str(upstream)},
        ]}))

    def run_preflight(self, *, offline="fail", online="ok", script=PREFLIGHT):
        environment = dict(os.environ)
        environment["PATH"] = f"{self.bin}:{environment['PATH']}"
        environment["STUB_LOG"] = str(self.log)
        environment["STUB_METADATA"] = str(self.metadata)
        environment["STUB_OFFLINE"] = offline
        environment["STUB_ONLINE"] = online
        return subprocess.run(
            ["bash", str(script), "--project", str(self.root),
             "--fork-manifest", str(self.fork), "--upstream-manifest", str(self.upstream),
             "--config", 'patch.crates-io.t4a-faer.path="x"'],
            capture_output=True, text=True, env=environment,
        )

    def calls(self):
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_an_offline_resolution_that_works_never_touches_the_network(self):
        result = self.run_preflight(offline="ok")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("retrying online", result.stderr)
        self.assertEqual(len(self.calls()), 1)
        self.assertIn("--offline", self.calls()[0])

    def test_a_cache_miss_prints_cargos_own_error_and_retries_online(self):
        result = self.run_preflight(offline="fail", online="ok")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("failed to download", result.stderr)
        self.assertIn("retrying online", result.stderr)
        self.assertEqual(len(self.calls()), 2)
        self.assertNotIn("--offline", self.calls()[1])

    def test_both_attempts_failing_is_reported_with_the_cause(self):
        result = self.run_preflight(offline="fail", online="fail")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed to download", result.stderr)
        self.assertIn("no route to crates.io", result.stderr)
        self.assertIn("cargo metadata failed", result.stderr)

    def test_a_checkout_that_is_not_the_resolved_one_is_refused_with_the_next_step(self):
        self.write_metadata(fork=self.root / "registry" / "t4a-faer" / "Cargo.toml",
                            upstream=self.upstream)
        result = self.run_preflight(offline="ok")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("would link t4a-faer from", result.stderr)
        self.assertIn("widening the range alone does not", result.stderr)

    def test_removing_the_online_retry_makes_this_suite_fail(self):
        """The regression the reviewer asked for: mutate the retry to `exit 1` and watch it fail."""
        mutated = self.root / "mutated_preflight.sh"
        source = PREFLIGHT.read_text()
        pattern = re.compile(
            r"if ! cargo metadata .*?--offline.*?\n(?:.*\n)*?fi\n\n(?=python3)", re.S)
        self.assertTrue(pattern.search(source), "the offline branch moved; update this mutation")
        mutated.write_text(pattern.sub("exit 1\n\n", source, count=1))
        result = self.run_preflight(offline="fail", online="ok", script=mutated)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("retrying online", result.stderr)


if __name__ == "__main__":
    unittest.main()
