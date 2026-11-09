# Cell terminal-row invariant

For every `(build, variant)` the parent must emit:

* exactly one `CELL start ... variant=<k>/<n>`;
* exactly one terminal CSV row (`ok`, `timeout` / `cut=timeout`, or `failed`), and exactly one
  matching `CELL done ... variant=<k> status=<...>`;
* a killed variant is the **only** one cut: the parent respawns a child for the remaining variants
  (`from=`), so they actually run. A variant the parent never reached is not reported as `timeout`.
* the variant count is known **before** construction (`variant_count`), so a build killed during
  construction still leaves one terminal row per variant.

The table is executed by `fbench/tests/child_cells.rs` with a fake child
(`fbench/tests/fake_child.sh`) that can finish immediately, hang while building/running, exit with
an error, or skip. The test asserts the number of table rows is exactly 8: adding an event without
adding its row (or the reverse) fails. `n = 3` is supplied through the test-only
`FBENCH_TEST_VARIANTS`, so first/middle/last are all covered.

| event | position | terminal rows (variants 1..3) |
|---|---|---|
| fast | all | ok, ok, ok |
| cut | first | timeout, ok, ok |
| cut | middle | ok, timeout, ok |
| cut | last | ok, ok, timeout |
| error | first | failed, ok, ok |
| error | middle | ok, failed, ok |
| error | last | ok, ok, failed |
| skip | all | skipped |

`cut` at first is also the construction-kill case: the child hangs before reporting any result for
variant 1, the parent cuts variant 1, then respawns at variant 2 and both 2 and 3 run (`ok`).
