# Guard stop note

This run completed with `rc=0`; the one-shot supervisor then stopped before its
data commit because the **global** post-run load exceeded 1.5.

- run: 20261113T124322Z-pub-325e482-20261113-mkl
- measured CPUs: 4-11 (fbench affinity verified there)
- observer/agents: 0-3,12-15
- guard: `/proc/loadavg` global, not measured-domain idle
- conclusion: the measurement is valid; this was a false stop, not a failed run.
- retry: none in the supervisor; the campaign was resumed manually with the
  completed runs skipped.
