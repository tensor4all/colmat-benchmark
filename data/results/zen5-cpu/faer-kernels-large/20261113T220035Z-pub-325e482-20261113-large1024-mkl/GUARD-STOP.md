# Guard stop note

This run completed with `rc=0`; the one-shot supervisor then stopped before its
data commit because the **global** post-run load exceeded 1.5.

- measured CPUs: 4-11
- guard: global /proc/loadavg, not measured-domain idle
- conclusion: the measurement is valid; this was a false stop.
- retry: the next step uses run-colmat-resumable.sh with a measured-CPU guard.
