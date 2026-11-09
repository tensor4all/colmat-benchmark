#!/usr/bin/env python3
"""Shim: the logic lives in campaign_toolkit.record_run.

Kept at this path, with the same CLI, so the campaign's docs, CI and habits do
not change when the shared tooling is updated.

It refuses a host the named hardware profile does not describe, before the
recorder builds, times or writes anything: `campaign_toolkit.record_run` compares
a profile with the CPU sets and the NUMA layout it declares, but never with the
machine, so a run taken on the 64-logical SMT-off host was recorded under
`zen2-cpu`, which declares 128 logical cores with SMT on (faer-benchmark#55).
The material is the one `idle_cpus.py check` reads -- the host's logical CPUs,
its SMT state and its sibling map against the profile's declaration -- and the
refusal prints both values. It also hands the profile name to the harness
(`FBENCH_PROFILE`), which writes it into the CSV's `profile` column.

Both live here, not in the shared package: the package is pinned by revision in
`requirements.txt`, so a package fix could not be pinned without merging it
first. The entry point is this script; `python -m campaign_toolkit.record_run`
still bypasses the check.
"""
import os
import pathlib
import sys

from campaign_toolkit import descriptor, hostinfo
from campaign_toolkit.record_run import main as record_run_main

CPU = "sys/devices/system/cpu"


def declared(profile):
    """(logical CPUs, SMT state, sibling offset) as the profile declares them.

    `undeclared` is not `off`: a profile without an `smt` block makes no claim
    about SMT (the toolkit's own default for thread rows), so there is nothing to
    compare it with.
    """
    smt = profile.get("smt") or {}
    per_core = smt.get("threads_per_core")
    return (profile["logical_cpus"], ("on" if per_core > 1 else "off") if per_core else "undeclared",
            smt.get("sibling_offset"))


def measured(root="/"):
    """(online logical CPUs, SMT state, {cpu: its live sibling set}) as this host reports them."""
    online = hostinfo.read(root, f"{CPU}/online")
    logical = (len(hostinfo.parse_cpu_list(online)) if online
               else len(hostinfo.logical_cpu_ids(root)))
    siblings = {}
    for cpu in hostinfo.logical_cpu_ids(root):
        text = hostinfo.read(root, f"{CPU}/cpu{cpu}/topology/thread_siblings_list")
        if text is not None:
            siblings[cpu] = hostinfo.parse_cpu_list(text)
    active = hostinfo.read(root, f"{CPU}/smt/active")
    if active in ("0", "1"):
        return logical, ("on" if active == "1" else "off"), siblings
    # No kernel SMT switch (an old kernel, a container, macOS): the sibling map says it.
    if siblings:
        return logical, ("on" if max(len(s) for s in siblings.values()) > 1 else "off"), siblings
    return logical, "unknown", siblings


def profile_siblings(profile, cpu):
    """CPU `cpu` and its siblings by the profile's `smt: {threads_per_core, sibling_offset}`."""
    smt = profile["smt"]
    return [cpu % smt["sibling_offset"] + k * smt["sibling_offset"]
            for k in range(smt["threads_per_core"])]


def require_host(profile, root="/"):
    """Exit non-zero unless this host is the machine `profile` declares."""
    logical, smt, siblings = measured(root)
    declared_logical, declared_smt, _ = declared(profile)
    if logical == 0:
        sys.exit(f"ERROR: cannot check this host against profile {profile['name']!r}: it reports no "
                 f"logical CPUs under {CPU}, so there is nothing to compare with the profile's "
                 f"declared {declared_logical} logical / smt {declared_smt}. Nothing was written.")
    if logical != declared_logical or (declared_smt != "undeclared" and smt != declared_smt):
        sys.exit(f"ERROR: this host is not the machine profile {profile['name']!r} declares: "
                 f"declared {declared_logical} logical / smt {declared_smt}, measured {logical} "
                 f"logical / smt {smt}. Recording here would file this machine's numbers under "
                 f"another machine; declare this host in benchmarks/profiles.yaml and record with "
                 f"that name. Nothing was written.")
    if declared_smt == "undeclared":
        return
    # Same comparison `idle_cpus.py check --profile` makes (its exit 4): the sibling map is
    # what makes one machine's CPU sets another machine's numbers.
    bad = [cpu for cpu, live in sorted(siblings.items())
           if cpu < declared_logical and sorted(live) != sorted(profile_siblings(profile, cpu))]
    if bad:
        cpu = bad[0]
        sys.exit(f"ERROR: this host's SMT layout is not the one profile {profile['name']!r} "
                 f"declares: cpu{cpu} has siblings "
                 f"{hostinfo.format_cpu_list(siblings[cpu])} here, the profile "
                 f"{hostinfo.format_cpu_list(profile_siblings(profile, cpu))} "
                 f"({len(bad)} of {len(siblings)} CPUs disagree). Nothing was written.")


def main(argv=None, root=None, host_root="/"):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent.parent)
    profiles = descriptor.set_root(root).profiles()
    # The profile is the recorder's first positional argument. Every declared profile named
    # anywhere in the arguments is a candidate, and two different ones are refused rather than
    # resolved: an option value (`--label`) that happens to name a profile would otherwise be
    # checked and exported in place of the profile the run records under.
    named = [token for token in argv if token in [p["name"] for p in profiles]]
    if len(set(named)) > 1:
        sys.exit(f"ERROR: the arguments name more than one hardware profile "
                 f"({', '.join(dict.fromkeys(named))}); a run is one profile, and an option value "
                 f"that matches a profile name makes the check ambiguous - rename it. Nothing was "
                 f"written.")
    profile = next((p for p in profiles if p["name"] == named[0]), None) if named else None
    if profile is not None:
        require_host(profile, host_root)
        # The recorder calls the harness with the static argv its descriptor declares, so the
        # profile reaches the harness through the environment it inherits.
        os.environ["FBENCH_PROFILE"] = profile["name"]
    return record_run_main(argv, root=root)


if __name__ == "__main__":
    sys.exit(main(root=pathlib.Path(__file__).resolve().parent.parent))
