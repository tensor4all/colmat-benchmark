# Reliable baseline re-record (timing policy v2)

The frozen contract for the baseline re-record and for candidate iterations is in
[protocol-v2-baseline.md](protocol-v2-baseline.md): settings (prime 250 ms, 5 calls,
3 process sets), the enforced machine preconditions (boost off, observer off the
measured L3 domains, whole-domain idle guard), what a recorded number means, and
the candidate-mode procedure. This file remains the reasoning behind it.

## Frozen identities

The optimization candidate is saved at fork `fe41bbdfb88c0528e0726dca3b1764f47222ad56`.
It is not the baseline and no optimization is adopted from the old scans.

The re-record uses exactly one clean checkout per library:

- `extern/colmat`: `bfe4041cd9918815e3b3f766ac0332b792badb7d`.
- `extern/upstream-faer`: `0539947ffb757a739d7e703a7d2fa0c792a909c1`.

Both vendor builds link these same paths/revisions. The faer adapters share a
macro body, fixtures, and the `Op` prepare/run/check timer. MKL and OpenBLAS are
separate binaries/runs because their symbols collide, not separate measurement
implementations. A comparison is about the declared public-API workload, not
proof of identical low-level algorithms or vendor internals.

V2 recording requires an **editable toolkit installation from its own clean Git
checkout**. The pinned editable VCS entry in `requirements.txt` retains that
checkout; local development can use `pip install -e ../campaign-toolkit` at the
same pinned commit. An ordinary installed distribution is refused rather than
attributing the enclosing harness repository's commit to the toolkit. Historical
v1 readers do not require this installation mode.

## Contract

Inputs match across providers for each `(family, dtype, shape, batch, order)`.
The shared flat seeded buffer is interpreted according to the requested order;
C/F are separate fixtures, not asserted to be the same logical matrix. A C/F
ratio is therefore not a pure layout experiment. Baseline/candidate comparisons
must keep the entire cell fixed.

| Family | Fixture salt | Actual retained result / check |
| --- | --- | --- |
| GEMM | A: `(m,k,n)`, B: `(k,n,m)` | caller C / reference product |
| factor-only QR | `min(m,n)` | packed factors + coefficients / reconstruct + orthogonality |
| explicit-Q QR | 3 | thin Q + R / reconstruct + orthogonality |
| partial / full-pivot LU | 5 / 7 | factors + permutations / reconstruct; vendor full-pivot skipped |
| Cholesky | 11, SPD | L / reconstruct |
| solve | A: 13, b: 17 | solution / residual; temporary factors die inside run |
| triangular solve | U: 19, b: 23 | solution / residual; vendor route skipped |
| singular values | 29 | spectrum / finiteness, ordering and Frobenius identity |
| thin SVD | 31 | U, S, V / reconstruct + orthogonality |
| self-adjoint EVD | 37, Hermitian | eigenvalues + vectors / residual + orthogonality |
| Householder apply | basis: 41, target: 43 | transformed target / alternate apply form |
| nonsymmetric EVD, factor upgrade | — | unimplemented, explicitly skipped |

The salt is the third argument of shared `seed(m,n,salt)` except the fully shown
GEMM arguments. Structured fixtures and batching are the same in both adapters.
Values-only SVD's invariant check is not a reference verification of every
singular value. Householder rows use basis width `min(m,n)`, target width `n`,
and one coefficient block; multi-panel sequences are not covered.

- GEMM, factor-only QR and Householder apply are retained-buffer calls: caller
  buffer initialization/restoration is outside the clock. Allocations internal
  to the provider API remain timed, including LAPACKE-owned workspace/adaptation.
- Factorization composites produce and retain actual outputs. The faer owning
  routes' public-API allocation/copy/factorization and explicit-Q construction
  are timed, and so is the destruction of the previous call's outputs: `prepare`
  only moves them aside and the next timed call frees them before it allocates
  (allocation and its matching free are both the library's cost). Solve
  consumes its temporary factor in the measured pipeline and retains the
  solution. What the harness allocates is not timed (next section).
- Checks inspect the actual last execution's factors/values/vectors/solution,
  not a replacement factorization of the original input. They do not check every
  timed call. Corrupting retained outputs must fail validation.
- The process owns one N-worker Rayon pool; both faer adapters use it, with
  `Par::Seq` for 1T. The vendor maximum budget is set/read back. Pool size and
  vendor budget are observed in the measured process and checked per row;
  neither is described as an active-worker trace.
- All chronological calls remain in `samples_ms`. V2 uses each process median,
  then the median across at least two complete-set process repeats (and complete
  engine-rotation cycles). Order remains a report dimension even if omitted from
  display-axis configuration. Reported between-process SD includes the
  rotating-order schedule; within-process SD remains separate.
  These are descriptive statistics, not confidence intervals or causal evidence.
- Verification emits a cell census, including separately timed variants in
  `note`. Every repeat must match it exactly: no missing/duplicate/failed cells,
  changed skip status, merged variants, missing processes, wrong rotation,
  nonfinite samples or inconsistent aggregates. Census base cells must match
  the declared population, dtype, order, shape, batch, engine and budget grid.
- Two checks, of two executions. The census (`fbench verify --threads t`) runs
  each cell once on the token the timed loop lends at that budget (a `colmat`
  team, the `faer` row's spindle group, the vendor's own threads), so the
  census at 4T and at 8T checks two different parallel executions. Until
  tensor4all/colmat-benchmark#127 the `colmat` census ran on `Token::none()`,
  which under the team API is sequential, so it never ran the path that was
  timed. Every timed cell is also checked separately: `measure_row` checks
  the state its **last timed repetition** left, after the row's reservation
  is released, and a failure makes the row `failed`. That check covers the
  measured call at every declared size, including sizes a narrowed census
  (`verify.n`) leaves out, but it records no residual.
- V1 artifacts/readers retain their historical interpretation. V2 pages use a
  timing-version suffix, so a new measurement contract does not overwrite the
  old numerical page. The exact clean toolkit commit is pinned and recorded as
  well as the harness and both library revisions.

## Allocation and the clock

**No primary row allocates, frees or converts layout inside its timed `run`** (rule from
`docs/measurement-audit-2026-11-06.md` fix 7, enforced by `fbench`'s `no_row_allocates_inside_run`
and `vendor_drivers_are_column_major_only`). One rule for every family and every arm, so a ratio
is never the difference between two harness choices:

- Every buffer is allocated before the clock, once per op, and reused by every priming and timed
  call: inputs and their references, the caller workspace a vendor driver works in (`LAPACKE_*_work`,
  sized by a `lwork = -1` query at op construction, plus `ipiv`/`tau`/`iwork`/`rwork`), the outputs
  a driver writes (`S`, `U`, `Vt`, eigenvalues, eigenvectors, the explicit `Q`), the faer in-place
  routes' matrix, right-hand side, permutations and `MemStack` (sized by the library's own
  `*_scratch`), and both arms' reference buffers.
- For `lu`, `cholesky`, `solve`, `qr`, `svd_values`, `svd_thin` and `eigh` (and `full_piv_lu`,
  `triangular_solve`, `householder_apply`, which have a vendor route since #131) every row makes
  the same kind of call: `prepare` restores the input into the prepared buffer, outside the clock, and the
  timed call is the decomposition (and solve, and output formation) alone -- `getrf_work`,
  `potrf_work`, `gesv_work`, `geqrf_work` + `ormqr_work`, `gesdd_work`, `syevd_work`/`heevd_work`,
  and for `full_piv_lu`, `triangular_solve` and `householder_apply` `?getc2`, `cblas_?trsm` and
  `?larfb_work`/`?ormqr_work` on the vendor row; on both faer rows one shared body
  (`define_backend!`) calling `lu_in_place`,
  `cholesky_in_place`, `solve_in_place_with_conj`, `full_piv_lu_in_place`, `qr_in_place` +
  `apply_block_householder_sequence_on_the_left_in_place`, `self_adjoint_evd` and the thin SVD's
  in-place route, each on the input's own layout. The harness converts nothing; a copy a *library*
  makes inside its call is its own cost (the fork's row-major LU factors into a column-major copy it
  allocates, `lu/partial_pivoting/factor.rs:396-418`), and it is never divided against a vendor row:
  a row-major cell of a vendored driver family is a `skipped` cell.
- The vendor row turns LAPACKE's input NaN scan off (`LAPACKE_set_nancheck(0)`, checked at startup),
  and a vendored driver family is measured column-major only: LAPACKE's row-major path allocates a
  transposed copy and transposes in and out of it inside the call in every `_work` variant, so a
  C-order cell of such a family is `skipped` with that reason instead of being charged a conversion
  the faer row does not pay. `gemm` keeps both orders (CBLAS's row-major entry point moves no data).
- The owning faer routes (`PartialPivLu::new`, `Llt::new`, `qr`'s owning Q and R, the `gesdd`/
  `syevd` wrappers, `partial_piv_lu().solve()` upstream, and the fork's composite rebuild of them)
  are kept as the call a user makes, but labelled **`api-inclusive`**: the library allocates the
  outputs and its scratch inside the timed call, so that row is API cost, not kernel cost. The
  joined page keys its cells on `note` (`KEY` in `scripts/gen_comparison.py`), so an `api-inclusive`
  row is never put in the same ratio as an empty-note row; it is published as a labelled secondary
  table and never headline a vendor ratio. The allocation-free rows carry an empty `note`.
- Because buffers are reused, `prepare` restores or spoils each retained result, so a check can only
  pass on what the timed call wrote. No case may depend on the allocator's adaptive behaviour: a
  measurement is not repaired with `MALLOC_*` or any other allocator setting, and a cell that moves
  with process history is traced to the allocation inside its clock. Numbers recorded before this
  rule are not comparable for these cells.

### Route differences that remain

- **`f32`/`c32`**: the vendor arm has the CBLAS routes in those two dtypes only -- `gemm`
  (`?gemm`) and `triangular_solve` (`?trsm`) -- and reports the LAPACK families as `skipped` for
  them, `full_piv_lu` (`?getc2`) and `householder_apply` (`?larfb`/`?ormqr`/`?unmqr`) included:
  the LAPACK drivers are bound for `f64`/`c64` only. The faer rows have no such gap.
- **`full_piv_lu`: the vendor route is unblocked `?getc2`, the faer rows' is the blocked
  `lu_in_place`.** No vendor ships a blocked complete-pivoting LU, so the vendor row is a different
  algorithm for the same contract (in place, `P A Q = L U`, prepared buffers), and the ratio is read
  with that. `?getc2` is square only: a non-square `full_piv_lu` cell is `skipped` on the vendor
  row with that reason. For `c64` the pivot metric differs too: `?zgetc2` picks the entry of
  largest modulus `|z|`, colmat and faer the largest `|re| + |im|`, so the two rows' pivot orders
  (and factors) differ while each is a complete-pivoting LU of the same matrix. LAPACKE does not
  wrap `?getc2`, so it is called through its Fortran symbol, which is column-major only: a c-order
  cell is `skipped` on the vendor row. The difference is declared in `fbench`
  (`lapack::ROUTE_DIFFERENCES`) and printed by `fbench info` as `route_difference.full_piv_lu=...`,
  which the recorder keeps in the run's `binary-info.txt`; the row's `note` stays empty, because
  the note is the join key and a different note would leave the cell without its vendor row.
- **`householder_apply`: the vendor applies colmat's reflectors.** `?larfb` (block form) and
  `?ormqr`/`?unmqr` (sequence form) apply the reflectors of colmat's own `qr_in_place` of the
  seeded matrix, converted to LAPACK's `H = I - tau v v^H` convention and with `?larft`'s `T`, all
  built once outside the clock. A complex LAPACK QR would normalise `R`'s diagonal to real where
  colmat keeps each pivot's phase, so the two rows would apply a different `Q`; with the same
  reflectors they compute the same `Q C`. The timed call is the vendor routine alone, on the
  colmat rows' notes (`block(..)`/`sequence(..)`), so block pairs with block and sequence with
  sequence.
- **A vendor route has to compute the colmat row's quantity**, not only pass its own residual
  check: `fbench`'s `vendor_routes_agree_with_colmat` runs each route on its op's input and compares
  the result with colmat's (`?trsm` and `?larfb`/`?ormqr`: the same numbers within the harness
  tolerance; `?getc2`: the same pivots and factors for `f64`, the same determinant for `c64`).
- **Families with no vendor route** (`upgrade_factor`, `eig`) are recorded as `skipped` with that
  reason and carry no `lapack-*` ratio. The difference is a property of the libraries, and the
  harness does not invent a vendor call to make the rows line up.
- **A route difference between arms is a property to record, not to fix away.** When two arms make
  different kinds of call, the page says so (note, skip reason, this section) and the ratio is read
  with it; the harness does not add or remove work on one arm until the numbers agree.
- **A page recorded before this rule does not print the ratios it could not measure fairly.** The
  joined view (`scripts/gen_comparison.py`) compares each arm's recorded `harness.commit` with
  `FAIR_HARNESS`, the harness that first charged both sides of a vendor ratio the same contract: at
  or after it the ratios are printed, before it (or when the checkout cannot place the commit) the
  vendor ratio cells of the families this contract changed say `withdrawn`, the numbers stay as
  recorded, and they are left out of the median. `FAIR_HARNESS` has to be a commit of `main`: a
  squash rewrites the SHAs of the landed change, so the constant is re-pinned to the landed commit
  when that happens (this campaign does both) and a test fails if it is not on `main`'s history. If
  it is ever orphaned the gate fails closed -- every vendor ratio withdrawn, never a one-sided one
  printed -- and the generator warns on stderr with the fix. The per-arm Markdown pages are the
  toolkit's render of their run and are not edited by hand; `docs/dataset-status.md` is their status
  of record.

## Adoption gate

First verify numerics and thread budgets, then examine representative A/A cells
covering small/alignment-sensitive/rectangular GEMM, factor and composite QR,
LU/solve, Cholesky/EVD/SVD, both dtypes/layouts, and 1/4/8T. Retain process starts
and execution order. Existing f64 GEMM evidence does not establish a universal
priming duration. Neither matching fork/upstream estimates nor selecting a
minimum is a validity criterion.

The proposed representative pilot uses both vendor binaries and C/F order, with
30 chronological calls, 3 complete process starts/row rotations, and a 3000 ms
priming floor. These are pilot settings, not yet suite defaults:

| Threads | Families | Shapes | Dtype |
| --- | --- | --- | --- |
| 1 | GEMM | 4, 256 | f64 |
| 1 | factor-only QR | 64x8, 8x64 | f64 |
| 4 | GEMM | 16x4096x240 | c64 |
| 8 | factor-only and explicit-Q QR | 4096x256 | f64 |
| 4 | LU, solve | 256 | c64 |
| 8 | Cholesky, EVD, values-only SVD | 512 | f64 |
| 4 | thin SVD | 64x1024 | c64 |
| 8 | Householder apply (both variants) | 4096x256 | f64 |

Correctness, declared/observed budgets and host guards must pass. The pinned
protocol's idle admission is an **admission** retry, not a data-selection device:
a run whose 3 s before/after windows were not idle is discarded, the process is
re-run on the same declared schedule, and every attempt is echoed in the guard
log; the retained CSV is the first attempt whose windows were idle. On this host
the operator/agent and unrelated background activity reject roughly 3.5% of
windows (5 rejections in ~142 measured windows, including two aborted one-attempt
control packages), so a one-attempt, one-shot rule cannot finish a multi-hour
package. Controls and the recorded suite therefore use `PINNED_RETRIES=8` with the
declared 3 s window, `max_busy=0.05` and 30 s wait, and the observing side
(driver, guard checker, agent) is launched on the CPUs outside the measured L3
domain - derived from the machine topology, not a fixed list. The recorder
**refuses to start** when its own affinity overlaps the measured set expanded with
SMT siblings, or when a declared measured set spans two L3 domains, so the pin is
an enforced precondition rather than a remembered habit; the agent's own pin lives
in its process and must be re-established after a restart. One representative
control (`data/probes/20261015-observer-pin/`) found the pin does not move the
measured distribution there; it removes a guard-rejection source and an
uncontrolled variable from the record. Increasing retries does not weaken the admission rule; it only decides how
many contaminated attempts are discarded before the package is abandoned.
Inspect per-cell process-median range relative to its median; flag a range above
5% as unstable, without discarding calls or replacing it with the minimum. This is
an explicit descriptive warning threshold, not a confidence bound. Record
within-process SD separately; an isolated retained outlier does not by itself
invalidate a median estimate.

Freeze the priming floor, calls/process, complete-set count, order schedule,
summary and noise/validity rules before the full suite. If representative cells
remain unstable, report that limitation rather than silently selecting/retrying
favorable results. The full data set records declared skips; it does not imply a
LAPACK equivalent exists for every faer operation.

Only after the full baseline data and distributions are available are the saved
optimizations reconsidered under the same fixed protocol.
