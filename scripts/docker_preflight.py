#!/usr/bin/env python3
"""Preflight of a Docker campaign (#99). Writes docs/preflight/docker-99-<UTC>.yaml and exits 1 unless
every check passes (`status: PASS`). Run on the host, from the observing CPUs, with the measured
container started (scripts/docker_campaign.sh start). Nothing is measured for the record here: the
per-arm smoke cell goes through the real path (recorder container -> pinned.sh -> docker exec) and
its run directory is moved aside, never published."""
import datetime, json, os, pathlib, shutil, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
IMAGE = os.environ.get("BENCH_IMAGE", "shinaoka/tenferro-bench-cpu:cpu-20261120-julia")
NAME = os.environ.get("BENCH_CONTAINER", "colmat-measured-99")
MEASURED, OBSERVER = "4-11", "0-3,12-15"
ENV = dict(os.environ, DOCKER_HOST="unix:///var/run/docker.sock")
checks = []


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, env=ENV, **kw)


def check(name, ok, detail=""):
    checks.append({"name": name, "status": "PASS" if ok else "FAIL", "detail": str(detail).strip()[:1500]})
    print(("PASS " if ok else "FAIL ") + name + (": " + str(detail).strip().splitlines()[0] if str(detail).strip() else ""))
    return ok


def allowed(pid_status):
    for line in pid_status.splitlines():
        if line.startswith("Cpus_allowed_list"):
            return line.split()[1]
    return None


# 1. the root daemon, pinned
info = sh(["docker", "info", "--format", "{{.SecurityOptions}}"])
check("docker daemon is not rootless", info.returncode == 0 and "rootless" not in info.stdout, info.stdout or info.stderr)
aff = sh(["systemctl", "show", "docker", "containerd", "-p", "CPUAffinity"]).stdout.split()
check("dockerd and containerd pinned to 0-3,12-15",
      aff.count("CPUAffinity=0-3") == 2 and aff.count("12-15") == 2, " ".join(aff))
# 2. the image
img = sh(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}} {{.RepoDigests}}"])
check("image present", img.returncode == 0, img.stdout or img.stderr)
# 3. the two containers
m = sh(["docker", "inspect", NAME, "--format", "{{.State.Running}} {{.HostConfig.CpusetCpus}} {{.Image}}"])
check("measured container running with cpuset 4-11", m.stdout.split()[:2] == ["true", MEASURED], m.stdout or m.stderr)
st = sh(["docker", "exec", NAME, "cat", "/proc/self/status"]).stdout
check("measured container sees only CPUs 4-11", allowed(st) == MEASURED, allowed(st))
rec = sh(["docker", "run", "--rm", f"--cpuset-cpus={OBSERVER}", "-v", "/usr/bin/docker:/usr/bin/docker:ro",
          "-v", "/var/run/docker.sock:/var/run/docker.sock", "-e", f"BENCH_CONTAINER={NAME}", IMAGE, "bash", "-c",
          f"grep Cpus_allowed_list /proc/self/status; docker exec {NAME} true && echo exec-ok"])
check("recorder container on 0-3,12-15 and can docker-exec into the measured one",
      allowed(rec.stdout) == OBSERVER and "exec-ok" in rec.stdout, rec.stdout + rec.stderr)
# 4. the host is quiet
idle = sh([sys.executable, str(ROOT / "benchmarks/scripts/idle_cpus.py"), "check", MEASURED, "--seconds", "10"])
check("measured domain idle (10 s, with SMT siblings and L3 peers)", idle.returncode == 0, idle.stdout + idle.stderr)
aud = sh([sys.executable, "-m", "campaign_toolkit.observer_audit", "--observer", OBSERVER])
check("no host process that used CPU may run on the measured side", aud.returncode == 0, aud.stdout + aud.stderr)
busy = sh(["pgrep", "-af", "fbench|record_run|pinned.sh|bench-cpu"])
other = [l for l in busy.stdout.splitlines() if NAME not in l and "docker_preflight" not in l and "pgrep" not in l]
ps = sh(["docker", "ps", "--format", "{{.Names}}"]).stdout.split()
check("no other benchmark process or container", not other and ps == [NAME], f"{other} {ps}")
power = {k: pathlib.Path(p).read_text().strip() if pathlib.Path(p).exists() else None for k, p in
         {"boost": "/sys/devices/system/cpu/cpufreq/boost",
          "governor_cpu4": "/sys/devices/system/cpu/cpu4/cpufreq/scaling_governor"}.items()}
# 5. the harness commit is the one the run will record
head = sh(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip()
dirty = sh(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]).stdout.strip()
pushed = sh(["git", "-C", str(ROOT), "branch", "-r", "--contains", head]).stdout.strip()
check("harness commit clean and on a remote", not dirty and bool(pushed), f"{head} {pushed} {dirty}")
# 6. a smoke cell per arm, through the real path
smokes = {}
for arm, engines in (("mkl", "faer-fork,faer-upstream,lapack-mkl"), ("openblas", "faer-fork,lapack-openblas"),
                     ("aocl", "faer-fork,lapack-aocl")):
    label = f"preflight-smoke-{arm}"
    r = sh([str(ROOT / "scripts/docker_campaign.sh"), "shell",
            f"python scripts/record_run.py zen5-cpu faer-kernels --vendor {arm} --engines {engines} "
            f"--regimes gemm-small --families gemm --threads 1,8 --dtypes f64 --reps 2 --aa {len(engines.split(','))} --prime-ms 250 "
            f"--no-publish --label {label}; chown -R {os.getuid()}:{os.getgid()} data .git docs result target"])
    runs = sorted((ROOT / "data/results/zen5-cpu/faer-kernels").glob(f"*-{label}"))
    detail = {"exit": r.returncode, "tail": r.stderr[-600:]}
    ok = r.returncode == 0 and len(runs) == 1
    if runs:
        run = runs[0]
        bi = dict(l.split("=", 1) for l in (run / "binary-info.txt").read_text().splitlines() if "=" in l) \
            if (run / "binary-info.txt").exists() else {}
        guard = "".join(g.read_text() for g in run.glob("*.guard"))
        in_container = f"measured in container {NAME}" in guard and "cpuset=4-11" in guard
        detail.update({"vendor.name": bi.get("vendor.name"), "vendor.version": bi.get("vendor.version"),
                       "guards_name_the_container": in_container})
        ok = ok and bi.get("vendor.name") == arm and in_container
        dest = ROOT / "target/preflight-smoke" / arm
        shutil.rmtree(dest, ignore_errors=True); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run), dest)
    smokes[arm] = detail
    check(f"smoke cell on the {arm} arm: builds, verifies, measures through the container", ok, json.dumps(detail))

status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
out = ROOT / "docs/preflight" / f"docker-99-{stamp}.yaml"
out.parent.mkdir(exist_ok=True)
lines = [f"status: {status}", f"issue: tensor4all/colmat-benchmark#99", f"timestamp: {stamp}", f"image: {IMAGE}",
         f"harness_commit: {head}", f"measured_container: {NAME}", f"power: {json.dumps(power)}", "checks:"]
for c in checks:
    lines += [f"  - name: {json.dumps(c['name'])}", f"    status: {c['status']}", f"    detail: {json.dumps(c['detail'])}"]
out.write_text("\n".join(lines) + "\n")
print(f"status: {status}  ({out})")
sys.exit(0 if status == "PASS" else 1)
