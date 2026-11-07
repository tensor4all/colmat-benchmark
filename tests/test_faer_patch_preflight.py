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
# Behave like the real cargo for the one argument this script must get right: --manifest-path
# takes a manifest file, so a directory has to be refused, not silently accepted by the stub.
manifest=""
prev=""
for arg in "$@"; do
    if [ "$prev" = "--manifest-path" ]; then manifest="$arg"; fi
    prev="$arg"
done
if [ -z "$manifest" ]; then
    echo "error: --manifest-path is required" >&2
    exit 1
fi
if [ -d "$manifest" ]; then
    echo "error: the manifest-path must be a path to a Cargo.toml file, not a directory: $manifest" >&2
    exit 101
fi
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
        self.manifest = self.root / "Cargo.toml"
        self.manifest.write_text("[workspace]\n")
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
            ["bash", str(script), "--manifest", str(self.manifest),
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
        # Rule 2 of test-double-fidelity: assert the recorded argv, not just "it returned".
        self.assertIn(f"--manifest-path {self.manifest}", self.calls()[0])
        # The patch arguments are the whole point of the preflight: if the caller stops forwarding
        # them, cargo resolves the registry crate and the comparison would compare the wrong thing.
        self.assertIn("--config patch.crates-io.t4a-faer.path=", self.calls()[0])

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

    def test_a_directory_given_as_the_manifest_is_refused_before_cargo_runs(self):
        result = subprocess.run(
            ["bash", str(PREFLIGHT), "--manifest", str(self.root),
             "--fork-manifest", str(self.fork), "--upstream-manifest", str(self.upstream)],
            capture_output=True, text=True, env={**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be a Cargo.toml path", result.stderr)
        self.assertFalse(self.log.exists(), "cargo was invoked with an unusable manifest path")

    def test_a_failure_that_is_not_a_download_is_not_retried(self):
        """A resolution error must not be dressed up as a network problem."""
        stub = self.bin / "cargo"
        stub.write_text(STUB.replace(
            'echo "error: failed to download t4a-gemm v0.20.1" >&2',
            'echo "error: failed to parse manifest" >&2'))
        result = self.run_preflight(offline="fail", online="ok")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not a download problem", result.stderr)
        self.assertNotIn("retrying online", result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_a_relative_manifest_is_accepted(self):
        """`Cargo.toml` is a path cargo accepts; the pattern must not demand a slash."""
        result = subprocess.run(
            ["bash", str(PREFLIGHT), "--manifest", "Cargo.toml",
             "--fork-manifest", str(self.fork), "--upstream-manifest", str(self.upstream),
             "--config", 'patch.crates-io.t4a-faer.path="x"'],
            capture_output=True, text=True, cwd=self.root,
            env={**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "STUB_LOG": str(self.log),
                 "STUB_METADATA": str(self.metadata)},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.calls())

    def test_a_stale_registry_index_is_retried(self):
        """`failed to select a version ... candidates found` means the local index is behind."""
        stub = self.bin / "cargo"
        stub.write_text(STUB.replace(
            'echo "error: failed to download t4a-gemm v0.20.1" >&2',
            'echo "error: failed to select a version for the requirement `demo = \\"=1.1.0\\"`" >&2; '
            'echo "candidate versions found which did not match: 1.0.0" >&2'))
        result = self.run_preflight(offline="fail", online="ok")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("retrying online", result.stderr)
        self.assertEqual(len(self.calls()), 2)

    def test_a_syntax_error_is_not_retried_even_when_the_path_says_network(self):
        """cargo's message can contain a directory named `network`; that is not a download failure."""
        stub = self.bin / "cargo"
        stub.write_text(STUB.replace(
            'echo "error: failed to download t4a-gemm v0.20.1" >&2',
            'echo "error: failed to parse manifest at /tmp/network/Cargo.toml: expected `]`" >&2'))
        result = self.run_preflight(offline="fail", online="ok")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("retrying online", result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_the_upstream_check_is_not_decorative(self):
        """Both measured crates are compared; only checking the fork would pass an upstream mix-up."""
        self.write_metadata(fork=self.fork, upstream=self.root / "wrong" / "faer" / "Cargo.toml")
        result = self.run_preflight(offline="ok")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("would link faer from", result.stderr)

    def test_the_caller_passes_a_manifest_not_a_directory(self):
        """The extraction made the call site an interface; that is the bug this test pins.

        `build_for_faer_rev.sh` used to hand `--project "$PROJECT_DIR"` to a flag that became
        `--manifest-path`, which real cargo rejects.  Running the whole build here would need a
        toolchain, so assert on the invocation the script writes.
        """
        caller = (REPO / "scripts" / "build_for_faer_rev.sh").read_text()
        self.assertIn('--manifest "$PROJECT_DIR/Cargo.toml"', caller)
        self.assertNotIn('--project "$PROJECT_DIR"', caller)

    def test_removing_the_online_retry_makes_this_suite_fail(self):
        """The regression the reviewer asked for: mutate the retry to `exit 1` and watch it fail."""
        mutated = self.root / "mutated_preflight.sh"
        source = PREFLIGHT.read_text()
        pattern = re.compile(
            r"if ! cargo metadata .*?--offline.*?\n(?:.*\n)*?fi\n\n(?=python3)", re.S)
        self.assertTrue(pattern.search(source), "the offline branch moved; update this mutation")
        mutated.write_text(pattern.sub("exit 1\n\n", source, count=1))
        result = self.run_preflight(offline="fail", online="ok", script=mutated)
        # The mutation has to break a *named* assertion, not merely exit non-zero:
        #   test_a_cache_miss_prints_cargos_own_error_and_retries_online asserts return code 0
        #   and "retrying online" in stderr.  Both are what this mutation destroys.
        self.assertNotEqual(result.returncode, 0, "the retry test's `returncode == 0` still holds")
        self.assertNotIn("retrying online", result.stderr,
                         "the retry test's `retrying online` assertion still holds")


if __name__ == "__main__":
    unittest.main()
