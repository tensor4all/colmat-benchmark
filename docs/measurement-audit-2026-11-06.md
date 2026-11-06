# Measurement audit, 2026-11-06

Read-only audit of what each timed closure does, traced to source. Nothing was run, no number
changed.

**Base:** repository `tensor4all/faer-benchmark` (local clone `/Users/hiroshi/projects/tensor4all/faer-benchmark`),
audited in a separate worktree at `origin/main` commit `c333615`, 2026-11-06. An earlier draft of this
document was written against a stale checkout (`75b8a1f`, 36 commits behind); every `file:line` below
was re-checked at `c333615`, and the diff `75b8a1f..c333615` in `fbench/src` was read in full: it
changes parallelism plumbing (a `Token` lent to each `run`, the fork row's `threadteam` team, `verify_par`),
not what is allocated, copied or clocked, so the findings are unchanged and only the line numbers moved.

Upstream library source read at tag `faer-v0.24.4` (`pins/upstream-faer.rev`, unchanged between the two
commits); reference LAPACKE read from `lapack-sys 0.15.0` in the cargo registry (the netlib wrapper
OpenBLAS bundles). **MKL's LAPACKE was not inspected**; every statement about what MKL does inside a
call is marked *unverified*.

Paths: `B` = `fbench/src/backend.rs`, `L` = `fbench/src/lapack.rs`, `T` = `fbench/src/timing.rs`,
`M` = `fbench/src/main.rs`. The fork row is `define_backend!(fork, …, explicit_ops)` (`B:2037`), the
upstream row is `define_backend!(upstream, …, highlevel_ops)` (`B:2038`).

## 0. Two findings that change the premise

1. **The vendor arm is not allocation-free in this source.** For `lu`, `cholesky`, `solve`,
   `svd_*`, `eigh` and `qr` (with Q) the closure copies the input with `to_vec()` inside the clock
   (`L:1018`, `L:1147`, `L:1235`, `L:1294`, `L:1363`, `L:1492`) and allocates `ipiv`/`sigma`/`w`
   (`L:1148`, `L:1364`, `L:1493`). Inside the call, reference LAPACKE adds a NaN scan
   (`lapacke_dpotrf.c:43-45`, `lapacke_dgetrf.c:43-45`), workspace `malloc`s for `dsyevd`/`dgesdd`
   (`lapacke_dsyevd.c:66,71`, `lapacke_dgesdd.c:59,72`), and for row-major input a malloc plus two
   transposes (`lapacke_dpotrf_work.c:55-68`, `lapacke_dgetrf_work.c:55-68`). The harness never calls
   `LAPACKE_set_nancheck` (no match in `fbench/src`). Same-layout copy and allocation are therefore
   inside *both* arms; what differs is the kind of tax (below). `qr_factor` is the exception: the
   vendor input is restored in `prepare` (`L:1007-1010`).
2. **The fork row does not run the library's high-level call; it runs a hand-written mirror of it**
   (`B:1259-1270` says so). Upstream's row calls upstream's own `Llt::new` / `PartialPivLu::new` /
   `split_LU`; the fork's row runs harness loops. They are not the same code, nothing tests that
   they cost the same, and the harness loops are scalar, bounds-checked element loops
   (`B:1282-1316`, `B:1639-1643`) where upstream uses `copy_from_triangular_*` and `z!` zips
   (`solvers.rs:955-980`, `:806-810`, faer-v0.24.4). See the evidence in section 4.

## 1. The timing loop, shared by every row

| item | where | verdict |
|---|---|---|
| Clocked region is exactly `op.run()`; `prepare` before, `check` once after all reps | `T:25-38` (clock `T:32-34`) | fine |
| Priming alternates `prepare`/`run` for `prime_ms` | `T:27-29` | fine, identical on every arm |
| Verification: `check` runs once on the last timed result, outside the clock; `verify` also runs before timing | `T:37`, `M:660` | fine |
| Faer rows hold their parallelism around priming, reps and check when threads > 1: the fork row a `threadteam` team (`faer_fork::team::with_team`, `M:161`, under `fork-team`), the upstream row the spindle group (`M:167`); vendor rows are not touched. The `verify` pass runs sequentially (`Token::none()`, `M:652-657`). | `M:152-172` | declared policy, outside the clock; a one-shot caller pays acquisition, the clock never sees it. Not the same mechanism on the two faer rows. |
| **Owning routes allocate inside `run` but their outputs are freed in `prepare`** (`outputs.clear()`, e.g. `B:1617`, `L:1230`) | all owning ops | **allocation timed, deallocation untimed.** Contradicts `docs/timing-policy.md:25-27` (v1 text: destruction is timed); v2 retains outputs. Allocator state then depends on size and history (see section 4). |
| Input generation, seeded fill, reference math | op constructors | outside, fine |

## 2. `fbench` (families in `backend.rs`, `lapack.rs`)

Parity = does the comparator make the same kind of call (owning vs in-place, alloc inside vs
outside, layout). "faer-vs-faer" is fork against upstream; "vs vendor" is against MKL/OpenBLAS.
Verification is outside the clock in every row.

| target | timed call | comparator parity | unfairly included | severity |
|---|---|---|---|---|
| `gemm` | fork/upstream `matmul(dst, Replace, …)` `B:361` into prepared `c`; vendor `T::gemm` `L:931` (`cblas_?gemm`, beta 0, `L:340,345`), `c` zeroed in `prepare` `B:353`/`L:926` | **same** | `parmod::mk(par)` per call, negligible | low; **fine** |
| `qr_factor` | fork+upstream share one body: `qrf::qr_in_place` `B:453`, matrix restored in `prepare` `B:440`, scratch `B:402-409`, `B:427` outside. Vendor: `geqrf` `L:1023`, input restored in `prepare` | **same** (in-place, buffers prepared) for the two faer rows. Vendor: in-place, but LAPACKE allocs workspace and, for C order, transposes inside the call (netlib `_work.c`; MKL *unverified*) | vendor only: LAPACKE workspace malloc + row-major transpose + NaN scan | medium: only for the vendor ratio. `qr_factor/n64` MKL is slower than `n96` in the same page section (0.2987 vs 0.2053 ms on `2240b076`; 0.2824 vs 0.2027 ms on `71b25ffb`), a non-monotone cell behind a published 3.3×. |
| `qr` (with Q) | upstream `mref().qr()` + `compute_thin_Q()` `B:768-770` (owning). Fork mirror `B:1384-1415`: `to_owned` (layout copy) `B:1384`, `coeff` zeros `B:1386`, `MemBuffer::new` `B:1398`, scalar `split_lu` `B:1401`, identity `B:1402`, second `MemBuffer::new` `B:1409`. Vendor `geqrf`+`orgqr` `L:1023,1046` after `to_vec` `L:1018`, `tau` alloc `L:1019`, `rmat` alloc+copy `L:1035-1041` | **different**: owning (fork, upstream) vs in-place-with-copy (vendor). The two faer rows also run different code (fork mirror vs library). | faer: alloc ×5, layout copy, split. Vendor: copy, tau, R copy, LAPACKE internals | medium (kernel is O(n³) at 30 µs and up) |
| `lu` | upstream `partial_piv_lu()` `B:833` = `PartialPivLu::new` (`to_owned`, perm vecs, scratch, `split_LU`). Fork `partial_lu_of` `B:1321-1341`: `to_owned` `B:1326`, 2 perm `Vec`s `B:1328-1329`, `MemBuffer::new` `B:1336`, scalar `split_lu` `B:1339`. Vendor `getrf` `L:1149` after `to_vec` `L:1147`, `ipiv` `L:1148` | **different** (owning + split vs in-place packed). faer rows also differ from each other (mirror vs library). | faer: layout copy, 3 allocs, L/U split (not part of an LU factorization as `getrf` defines it). Vendor: copy, `ipiv`, NaN scan, C-order transposes | **high** (n ≤ 96, see section 4) |
| `full_piv_lu` | upstream `B:884`; fork `B:1537-1560` (`to_owned`, 4 perm vecs, scratch, `split_lu`). No vendor route (`L:873-875`, skipped) | fork vs upstream: **different** (mirror vs library). Vendor: not compared. | owning tax shared by both faer rows, but the fork's copy is the harness's scalar loop | **high** for the fork-vs-upstream ratio (n ≤ 96) |
| `cholesky` | upstream `.llt(Side::Lower)` `B:934-936` = `Llt::new`: `Mat::zeros`, `copy_from_triangular_lower`, `MemBuffer::new`, factor, zero upper (`solvers.rs:773-818`). Fork `B:1624-1644`: `zeros` `B:1624`, `copy_from_triangular_lower` `B:1625`, `MemBuffer::new` `B:1635`, `cholesky_in_place` `B:1631`, scalar upper-zero loop `B:1639-1643`, push `B:1644`. Vendor `potrf` `L:1236` after `to_vec` `L:1235` | **different** (owning, allocating, layout-copying vs copy + LAPACKE-internal). Fork vs upstream also differ in the zero loop. | faer: n×n alloc, triangle copy that transposes a C-order input, scratch alloc, upper zeroing. Vendor: full copy, NaN scan, C-order malloc + 2 transposes | **high** (n ≤ 128; this is the known case) |
| `solve` | upstream `partial_piv_lu()` `B:985` + `solve_in_place` `B:987`. Fork `partial_lu_of` `B:1711` + `plu_solve::solve_in_place_with_conj` `B:1713` with `MemBuffer::new` `B:1720`. Vendor `gesv` `L:1296` after `to_vec` `L:1294`, `ipiv` `L:1295` | **different**: faer factors to separate L and U then solves; `gesv` keeps packed LU | faer: everything under `lu` plus a second scratch alloc. Vendor: copy, `ipiv`, C-order transposes | **high** (n ≤ 96) |
| `triangular_solve` | upstream `solve_upper_triangular_in_place` `B:1044`; fork `B:1782`; rhs restored in `prepare` `B:1037`/`B:1775`. No vendor route | **same** (in-place, no allocation) | `acc(&mut sink, …)` one element, negligible | low; **fine** |
| `svd_values` | upstream `singular_values()` `B:1099` (`Diag::zeros`, `MemBuffer::new`, `svd`, collect: `solvers.rs:457-481`). Fork `B:1843-1862`: same steps by hand. Vendor `gesdd` JOB_N `L:1378` after `to_vec` `L:1363`, `sigma` `L:1364` | faer rows: same kind. vs vendor: **different** (faer `svd` reads `A` without a copy; vendor copies it and LAPACKE mallocs work) | faer: S alloc + scratch alloc. Vendor: copy, S, work malloc, NaN scan | low (kernel ≥ 64 µs at n = 32; the vendor tax runs against the vendor, so a "fork slower than MKL" claim is conservative) |
| `svd_thin` | upstream `thin_svd()` `B:1177`; fork `B:1917-1937`. Vendor `gesdd` JOB_S `L:1378` | same pattern as `svd_values`; U, V/VT allocs are returned outputs | as above | low |
| `eigh` | upstream `self_adjoint_eigen(Lower)` `B:1231`; fork `B:1996-2012`. Vendor `syevd`/`heevd` `L:1494` after `to_vec` `L:1492`, `w` `L:1493` | faer rows: same kind. vs vendor: **different** (as `svd`) | as above; `dsyevd` does workspace query + 2 mallocs (`lapacke_dsyevd.c:66,71`) | low |
| `householder_apply` (block, sequence) | `apply_block_householder_*_on_the_left_in_place_with_conj` `B:623-627`; QR fixture, `c` restore and scratch (`s.mem`, `B:568-572`, `B:594`) all outside. No vendor route | **same** (in-place, scratch prepared) for the two faer rows | none | low; **fine** |
| `upgrade_factor`, `eig` | not implemented, recorded `skipped` (`B:306-308`, `M:52-58`) | n/a | n/a | n/a |

## 3. Other targets

`tcbench`, `lukbench`, `exec_entry`, `contract` and `capi_rust` are **not in this repository**
(`fbench` is the only workspace member, `Cargo.toml`; no file of those names, no `src/` outside
`fbench`). Textual matches for `tcbench`, `lukbench` and `capi_rust` exist only under the sibling
checkout `tcontract-rs-wt32/benchmarks/` and `.../crates/`; `exec_entry` appears only in
`strided-rs-*/docs/design/view-metadata-inline.md`. They were not audited, because tracing them
would be tracing another repository's source. Tell me which checkout is canonical and I will audit
it the same way.

## 4. Evidence from the published pages (no new run)

Newest page: `result/zen5-cpu/faer-kernels/71b25ffb7ae3-0539947ffb75-mkl-timing-v2.md` (fork `71b25ffb`,
`t4a-faer 0.27.1`; harness `eeeaa36`, an ancestor of `c333615`; development host, provisional), section
`linalg-small, f64, c order, 1T` (page line 482). Read as a consistency check, not as proof.

| family | n = 32 / 64 / 96, fork − upstream (µs) | per n² (ns) | reading |
|---|---|---|---|
| `lu` | 1.42 / 4.74 / 6.58 | 1.4 / 1.2 / 0.7 | roughly quadratic, noisy at 96 |
| `solve` | 1.65 / 4.63 / 13.50 | 1.6 / 1.1 / 1.5 | roughly quadratic |
| `full_piv_lu` | 2.17 / 6.80 / 12.48 | 2.1 / 1.7 / 1.4 | roughly quadratic |
| `cholesky` | −0.22 / 0.82 / 1.13 | −0.2 / 0.2 / 0.1 | gap gone (fork now faster at n = 32) |

The previous page (`2240b076…`, fork `0.24.4`) showed the same three families with a cleaner n² scaling
(`lu` 1.2 / 4.4 / 10.4 µs, ≈1.1 ns per element) and a `cholesky` gap of +0.28 / +1.69 / +1.84 µs.

A fork-minus-upstream gap that grows like n², about 1 ns per element, is what the harness's scalar split
and zero loops (`B:1282-1316`, `B:1639-1643`) would produce, and is not what an O(n³) kernel difference
looks like. It vanishes where the kernel dominates (at `lu/n128`, fork 0.1338 vs upstream 0.1363 ms on the
older page). The library may differ too; the page cannot separate the two. The cheap test is to time
`split_lu` against upstream's `split_LU` alone, or to run `lu` through in-place calls on both rows
(section 5, item 2). On the newer page `cholesky` no longer shows a "fork slower" gap, so it stays in this
audit for the structural reasons (owning call, mirror code) but not as an observed fork-vs-upstream loss.

Allocator mechanism, a hypothesis not measured here: an n×n `f64` matrix crosses glibc's default 128 KiB
mmap threshold at n = 128 (c64: n = 96). The first allocation at or above it is `mmap`ed (page faults
inside the clock); a `free` of an `mmap`ed chunk raises the threshold, so later allocations come from the
heap. Because the free happens in `prepare`, outside the clock, the same cell can move 2× between runs
depending on when the threshold shifted. This matches the `cholesky` behaviour reported.

Side note from `B:1318-1320`: before `0e685d7` (#37) the fork `lu` and `solve` rows ran their timed
factorization on `Par::Seq` at every thread count under `fork-team`. Pages recorded with a harness
between #35 and #37 would carry that; which pages those are was not determined here. The two pages
above predate #35.

## 5. Ranked fixes (most threatened published number first)

1. **`cholesky`, fork + upstream: owning (`Llt::new` / mirror) → `in_place`.** Allocate the matrix and
   the `MemStack` (sized by `cholesky_in_place_scratch`) before the clock, restore the matrix from
   `a0` in `prepare`, call `cholesky_in_place` on a `MatMut` over the prepared buffer inside it. No
   zero-upper loop in the clock (check reads the lower triangle). Same change on the vendor row:
   `to_vec` out of `L:1235`, restore in `prepare`, `potrf` on the prepared buffer.
2. **`lu`, fork + upstream: owning (`partial_piv_lu` / `partial_lu_of`) → `lu_in_place`** on the
   prepared buffer with prepared `fwd`/`bwd` and scratch; drop the L/U split (`getrf` has none; the
   check can split outside, as `B:1476-1498` already reconstructs outside). Vendor: restore in
   `prepare`, `ipiv` outside.
3. **`solve`: owning → `lu_in_place` + a solve over the packed factor** (faer's
   `solve_in_place_with_conj` takes `L` and `U`; passing the packed matrix as both should work if the
   triangular solves ignore the other triangle: *to be confirmed in the fork*), with `MemStack`
   prepared. Vendor: `gesv` on a prepared buffer.
4. **`full_piv_lu`: owning → `flu::lu_in_place`** on a prepared buffer, four perm vecs and scratch
   outside, no split. No vendor row; only the fork-vs-upstream ratio is at stake.
5. **`qr` (with Q): decide the contract first.** It returns explicit Q and R, so an owning call is
   legitimate; then both vendor and faer must pay the same: prepared output buffers for Q and R on
   both sides, `qr_in_place` + apply into prepared Q, a single prepared `MemStack`.
6. **Run fork and upstream through the same harness code.** Replace the fork mirror
   (`explicit_ops!`) and the upstream high-level calls (`highlevel_ops!`) for the in-place families
   with one body, so a fork-vs-upstream ratio cannot contain a harness difference. Today only
   `gemm` and `qr_factor` (`define_backend!`) and `householder_apply` are single-source.
7. **`svd_values`, `svd_thin`, `eigh`: prepared outputs and scratch.** `svd`/`self_adjoint_evd` take
   `A` read-only, so no input copy is needed on the faer side: preallocate `S`, `U`, `V` and the
   `MemBuffer` (`B:1843-1858`, `B:1917-1934`, `B:1996-2009`) and call straight on `a0`. Vendor still
   needs a restore in `prepare` (`gesdd`/`syevd` overwrite `A`).
8. **Free outputs inside the clock or allocate none.** With 1-7 done no owning route remains in the
   factor/solve/decompose families; for `qr`, either time the drop or retain nothing.
9. **Disable the LAPACKE NaN scan on the vendor arm** (`LAPACKE_set_nancheck(0)`) or record it as
   timed; today the harness neither disables nor records it.
10. **Document the residual vendor row-major cost** (LAPACKE malloc + transposes for C order) as
    an intrinsic route difference, as `faer-kernels.yaml:88` already says, and state that it is not
    present for F order.

`in_place` vs `owning`, per family, as of today:

| family | fork | upstream | vendor |
|---|---|---|---|
| gemm | in-place | in-place | in-place |
| qr_factor | in-place | in-place | in-place (+LAPACKE internals) |
| qr | owning | owning | copy + in-place |
| lu | **owning** | **owning** | copy + in-place |
| full_piv_lu | **owning** | **owning** | none |
| cholesky | **owning** | **owning** | copy + in-place |
| solve | **owning** | **owning** | copy + in-place |
| triangular_solve | in-place | in-place | none |
| svd_values / svd_thin / eigh | owning outputs + scratch alloc | owning (library call) | copy + in-place |
| householder_apply | in-place | in-place | none |

## 6. Cells that cannot be defended until the fix lands

All traced to the calls in section 2. Values from the newest page of section 4 (`71b25ffb`, c order,
1T, `linalg-small`) unless noted; the same calls produce the other orders, thread counts and dtypes, but
those values were not read number by number.

- **Fork vs upstream, `lu/n32,n64,n96`, `solve/n32,n64,n96`, `full_piv_lu/n32,n64,n96`** (published 1.10-1.21×
  slower): `B:1282-1316` and `B:1321-1341` against `solvers.rs:955-980`; the n² scaling in section 4 says
  these are mostly harness.
- **Fork vs upstream, `cholesky/n32,n64,n96`**: the fork row runs the zero loop at `B:1639-1643`, which
  upstream's row does not. The newest page reads 1.07× *faster* at n = 32, the older one 1.07-1.11× slower;
  neither can be read as a library difference until both rows run one body.
- **Fork vs MKL, `lu/n32..n128`, `solve/n32..n128`, `cholesky/n32..n128`** (published 1.2-1.65× faster):
  the fork pays an owning tax and MKL pays a copy plus LAPACKE internals; the sign of the net error
  cannot be determined from source.
- **Fork vs MKL, `qr_factor/n64`** (published 3.30× faster, page line 508): MKL reads 0.2824 ms ± 0.124
  against 0.2027 ms at `n96` in the same section, non-monotone with a large between-start spread. `L:1023`
  is the call; the harness restores the input outside, so the cause is not visible in source. Re-take.
- **Fork vs upstream `qr_factor`** (1.19-1.58× faster): both rows run the same in-place body
  (`B:453`), so this audit finds no harness difference. Not withdrawn.
- **`svd_*`, `eigh` vs MKL**: the harness tax (copy, work malloc) runs against MKL, so "fork slower"
  claims are conservative. Not withdrawn, but their ratios should still be re-taken after fix 7.
- **Not withdrawn, fine as traced:** `gemm`, `triangular_solve`, `householder_apply`. (`triangular_solve`
  reads 1.06-1.07× slower at n = 64/96 on the newest page; both rows are the same in-place call
  `B:1044`/`B:1782`, so that is not a harness difference this audit can see.)

I did not read the other orders, thread counts or dtypes number by number; the trace applies to those
cells too, but their published values were not checked here.
