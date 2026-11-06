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

- `extern/t4a-faer`: `bfe4041cd9918815e3b3f766ac0332b792badb7d`.
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
- V1 artifacts/readers retain their historical interpretation. V2 pages use a
  timing-version suffix, so a new measurement contract does not overwrite the
  old numerical page. The exact clean toolkit commit is pinned and recorded as
  well as the harness and both library revisions.

## Allocation and the clock

Every buffer the harness owns is allocated before the clock, once per op, and reused by every
priming and timed call: inputs and their references, the caller buffers a vendor routine works in
(`ipiv`, `tau`, singular values and vectors, eigenvalues, the explicit `R`), the faer `in_place`
routes' matrix, right-hand side, permutations and `MemStack` (sized by the library's own
`*_scratch`), and the comparison arms' buffers. For `lu`, `cholesky` and `solve` every row makes
the same kind of call: `prepare` restores the input into the prepared buffer, outside the clock,
and the timed call is the in-place factorization (and solve) alone - `getrf`/`potrf`/`gesv` on
the vendor row, and on both faer rows one shared body (`define_backend!`, noted `in_place`) calling
`lu_in_place`, `cholesky_in_place` and `solve_in_place_with_conj` on the input's own layout. The
vendor row turns LAPACKE's input NaN scan off (`LAPACKE_set_nancheck(0)`, checked at startup);
what LAPACKE still does for itself, such as a row-major transpose and its workspace, is the
vendor's cost and stays inside. The owning faer routes (the op with an empty `note`:
`PartialPivLu::new`, `Llt::new`, `partial_piv_lu().solve()` upstream and the fork's composite
rebuild of them) are kept as the call a user makes; what they allocate is the library's cost and is
never hoisted, so those cells move with the allocator's state (a 16 MB heap trim at n = 1024 is
about 3 ms) and are not the comparison with the vendor row; their previous outputs are freed
inside the clock, so a call is charged for the free that its allocation implies. The faer
`in_place` routes exist for `full_piv_lu`, `svd_values`, `svd_thin` and `eigh` as well, written
once in `define_backend!` for both faer rows: the matrix and permutations (`full_piv_lu`) or the
outputs `S`, `U`, `V` (`svd`, `eigh`) and the `MemStack` sized by the library's own `*_scratch` are
allocated before the clock, and the decompositions run straight on the input, which they read
without modifying. The vendor row's `gesdd` and `syevd`/`heevd` overwrite their input, so `prepare`
restores it into a prepared buffer and the timed call is the driver alone. No row keeps an input
copy inside the clock except an owning faer route, whose copy is the library's. Because buffers are
reused, `prepare` restores or spoils each
retained result, so a check can only pass on what the timed call wrote. No case may depend on the
allocator's adaptive behaviour: a measurement is not repaired with `MALLOC_*` or any other
allocator setting, and a cell that moves with process history is traced to the allocation inside
its clock. Numbers recorded before this rule are not comparable for these cells.

### Route differences that remain

- **`qr` (explicit Q and R) has no vendor row.** The faer rows return an owning Q and R, which is what
  the family declares and a legitimate contract; a vendor row can be made to pay the same only by
  preparing Q and R on one side and letting the library allocate on the other, which compares two
  harness choices rather than two libraries. The decision is to record the difference, not to bend an
  arm: the vendor row reports `qr` as `skipped` with that reason, no `qr` ratio against `lapack-*`
  is claimed, and `qr_factor` (`geqrf` against `qr_in_place`, both on prepared buffers) is the
  like-for-like QR comparison. The two faer rows are still compared with each other on `qr`, both
  owning.
- **Row-major input on the vendor row.** For C order, LAPACKE allocates a work copy and transposes in and
  out of it inside the call (reference `_work.c`; MKL's LAPACKE was not inspected, so that is unverified
  for MKL). That cost is intrinsic to the row-major entry point, stays inside the clock and is not
  present for F order. A C-order vendor ratio therefore includes a cost the F-order one does not; the
  faer rows factor a row-major input where it is.
- **A route difference between arms is a property to record, not to fix away.** When two arms make
  different kinds of call, the page says so (note, skip reason, this section) and the ratio is read
  with it; the harness does not add or remove work on one arm until the numbers agree.

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
