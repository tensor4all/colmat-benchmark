#!/usr/bin/env bash
# Build the campaign harness (`fbench`) with both faer libraries linked *by path*.
#
#   build_for_faer_rev.sh --checkout <fork-dir> --upstream <upstream-dir> \
#                         --vendor <openblas|mkl> --out <var-file> [--bin fbench]...
#
# Writes shell assignments to <var-file>:
#   FAER_FORK_REV, FAER_FORK_DIRTY, FAER_FORK_DIR, FAER_FORK_VERSION, FAER_FORK_FEATURES,
#   FAER_UPSTREAM_REV, FAER_UPSTREAM_DIRTY, FAER_UPSTREAM_DIR,
#   FAER_UPSTREAM_VERSION, FAER_UPSTREAM_REF, FAER_UPSTREAM_FEATURES,
#   BIN_DIR, BUILD_FEATURES, BUILD_ENV, BUILD_VENDOR
#
# The measured library is the fork checkout named by `--checkout`; the baseline is
# the upstream checkout named by `--upstream`. Neither is fetched from a registry:
# the harness links both by path, so a run's manifest can name the two revisions
# it actually measured. `fbench` is built from *this* repository, because the
# harness is not part of the fork (the fork's `AGENTS.md` forbids keeping campaign
# infrastructure there, since it would be a rebase cost).
#
# Three things this deliberately does *not* do.
#
# It does not let baseline and candidate share a target directory, and it clears
# the compiler cache for the build: a cache keyed on inputs that do not include
# the sibling checkout can mix revisions. The target directory is keyed by the
# fork revision *and* the vendor arm, because the two arms are different feature
# sets and a shared directory would have the second build replace the first
# binary while the manifest still named the first.
#
# It does not let the environment decide the vendor's thread count. The build is
# asked for the vendor statically where it can be (`link-openblas-static` is the
# sibling's choice; this harness declares `link-openblas`), so the recorder can
# execute the binary directly instead of arranging a loader search path;
# `OPENBLAS_NUM_THREADS` is dropped because `openblas-src` folds it into the built
# library, and the thread budget belongs to the measurement, not to the build.
# `OPENBLAS_DYNAMIC_ARCH` is *set* below, because that is the declaration of how
# the vendor was built.
#
# It does not guess which MKL either. `--vendor` names the arm -- one binary links
# one vendor library, so an arm is a separate build into a separate target
# directory -- and the MKL arm's `MKLROOT` and `MKL_THREADING_LIB` are resolved to
# *versioned* paths and reported back in `BUILD_ENV`. Nothing here follows
# `/opt/intel/oneapi/mkl/latest`: a moving symlink let a build record a version it
# never read back.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
    echo "usage: build_for_faer_rev.sh --checkout <fork-dir> --upstream <upstream-dir>" >&2
    echo "                            --vendor <openblas|mkl> --out <var-file> [--bin <name>]..." >&2
}

FORK_DIR=""
UPSTREAM_DIR=""
OUT_FILE=""
VENDOR="${BENCH_VENDOR:-}"
PROFILE="${BENCH_BUILD_PROFILE:-release}"
BINS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkout) FORK_DIR="${2:?--checkout needs a path}"; shift 2 ;;
        --upstream) UPSTREAM_DIR="${2:?--upstream needs a path}"; shift 2 ;;
        --vendor) VENDOR="${2:?--vendor needs a name}"; shift 2 ;;
        --out) OUT_FILE="${2:?--out needs a file}"; shift 2 ;;
        --bin) BINS+=("${2:?--bin needs a name}"); shift 2 ;;
        --profile) PROFILE="${2:?--profile needs a name}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument '$1'" >&2; usage; exit 2 ;;
    esac
done
if [[ -z "$FORK_DIR" || -z "$UPSTREAM_DIR" || -z "$OUT_FILE" || -z "$VENDOR" ]]; then
    echo "ERROR: --checkout, --upstream, --vendor and --out are all required" >&2
    usage
    exit 2
fi
[[ ${#BINS[@]} -gt 0 ]] || BINS=(fbench)

real="$(cd "$FORK_DIR" 2>/dev/null && pwd -P)" || { echo "ERROR: --checkout '$FORK_DIR' is not a directory" >&2; exit 1; }
FORK_DIR="$real"
real="$(cd "$UPSTREAM_DIR" 2>/dev/null && pwd -P)" || { echo "ERROR: --upstream '$UPSTREAM_DIR' is not a directory" >&2; exit 1; }
UPSTREAM_DIR="$real"
[[ -d "$FORK_DIR/.git" ]] || { echo "ERROR: $FORK_DIR is not a git checkout" >&2; exit 1; }
[[ -d "$UPSTREAM_DIR/.git" ]] || { echo "ERROR: $UPSTREAM_DIR is not a git checkout" >&2; exit 1; }
# The fork publishes the crate as `t4a-faer` and upstream calls it `faer`, but both carry the
# package in their `faer/` workspace member. A checkout without it is not a faer tree.
[[ -f "$FORK_DIR/faer/Cargo.toml" ]] || { echo "ERROR: $FORK_DIR/faer/Cargo.toml not found; not a faer checkout" >&2; exit 1; }
[[ -f "$UPSTREAM_DIR/faer/Cargo.toml" ]] || { echo "ERROR: $UPSTREAM_DIR/faer/Cargo.toml not found; not an upstream faer checkout" >&2; exit 1; }
[[ -f "$PROJECT_DIR/fbench/Cargo.toml" ]] || { echo "ERROR: $PROJECT_DIR/fbench/Cargo.toml not found; the harness crate has not been written yet" >&2; exit 1; }

fork_rev="$(git -C "$FORK_DIR" rev-parse HEAD)"
up_rev="$(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
fork_dirty=false
[[ -z "$(git -C "$FORK_DIR" status --porcelain --untracked-files=no)" ]] || fork_dirty=true
up_dirty=false
[[ -z "$(git -C "$UPSTREAM_DIR" status --porcelain --untracked-files=no)" ]] || up_dirty=true
suffix=""
[[ $fork_dirty == true || $up_dirty == true ]] && suffix="-dirty"

# The package version of each crate, read from its own manifest rather than assumed from a pin.
version_of() {
    python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb")).get("package", {}).get("version", ""))' "$1"
}
# Library feature requests are not the harness's vendor-link feature selector.
features_of() {
    python3 -c 'import sys, tomllib; print(",".join(tomllib.load(open(sys.argv[1], "rb"))["dependencies"][sys.argv[2]].get("features", [])))' "$PROJECT_DIR/fbench/Cargo.toml" "$1"
}
fork_version="$(version_of "$FORK_DIR/faer/Cargo.toml")"
up_version="$(version_of "$UPSTREAM_DIR/faer/Cargo.toml")"
# The tag the upstream checkout sits on, when it sits on one; informational, so a pin is readable
# as a release and not only as a hash.
up_ref="$(git -C "$UPSTREAM_DIR" describe --tags --exact-match 2>/dev/null || git -C "$UPSTREAM_DIR" describe --tags --abbrev=0 2>/dev/null || true)"

# Keyed by the fork revision and the vendor as well as the dirty state: the two arms are different
# feature sets, and a shared target directory would have the second build replace the first binary
# while the manifest still named the first.
target="$PROJECT_DIR/target/faer-rev/${fork_rev:0:12}-${VENDOR}${suffix}"
profile_flag=(); subdir=debug
if [[ "$PROFILE" == release ]]; then profile_flag=(--release); subdir=release; fi

# The glob is passed quoted, so the shell does not expand it to many arguments and this
# function sees only the first: with two installations the newest must still win.
newest_dir() { ls -d $1 2>/dev/null | sort -V | tail -1; }

case "$VENDOR" in
  openblas)
    FEATURES="fbench/link-openblas"
    # Declared, not inherited. `openblas-src` reads `OPENBLAS_DYNAMIC_ARCH` at build time; without
    # it the from-source OpenBLAS guesses a target from the build host (here `COOPERLAKE`, an
    # Intel AVX-512 target), and a guessed target is not one the run-time CPU is obliged to
    # execute -- the sibling campaign's linked tests died with SIGILL on CI exactly that way.
    # `DYNAMIC_ARCH=1` compiles several cores and dispatches on CPUID with the required features
    # checked per core; on this host that dispatcher still selects COOPERLAKE, and the manifest
    # records the kernel it dispatched to.
    #
    # OpenBLAS's LAPACK is Fortran, so the build needs a Fortran compiler, and which one built
    # the vendor library is part of its identity. `gfortran` is the first choice; where it is not
    # installed the oneAPI Fortran compiler is, at a versioned path, and OpenBLAS builds against
    # Intel Fortran as well. Whichever is used is put on `PATH` for the build and recorded in
    # `BUILD_ENV`, because a page that names a vendor library has to say what compiled it.
    if command -v gfortran >/dev/null; then
      OPENBLAS_FC="$(command -v gfortran)"
    else
      OPENBLAS_FC="$(newest_dir "/opt/intel/oneapi/compiler/*/bin/ifx")"
      [[ -n "$OPENBLAS_FC" ]] || {
        echo "ERROR: no Fortran compiler for the OpenBLAS source build: neither gfortran nor" \
             "oneAPI ifx was found; install one or record the MKL arm only" >&2; exit 1; }
    fi
    export FC="$OPENBLAS_FC"
    # `FC` alone is not enough for an Intel front end: OpenBLAS's own Makefile picks its
    # `F_COMPILER` from `FC`'s basename, and the oneAPI runtime has to be reachable at run time
    # for the build's own test programs.
    OPENBLAS_FC="$(readlink -f "$OPENBLAS_FC")"
    # The build runs OpenBLAS's own test programs, and the Fortran runtime they link is loaded
    # dynamically: without the directory it lives in on the loader path they die with
    # "error while loading shared libraries: libimf.so" and the build is reported as failed
    # *after* it has compiled the whole library. The loader path is recorded beside the compiler
    # for the same reason the compiler is: that pair is what the vendor library's LAPACK was
    # built and checked with.
    OPENBLAS_FC_LIB="$(cd "$(dirname "$OPENBLAS_FC")/../lib" && pwd -P)"
    # The Fortran runtime is installed in two places under oneAPI -- beside the compiler and under
    # the oneAPI version directory -- and the binary needs both on its search path. Both are
    # derived from the compiler's own path rather than guessed from one installation's layout.
    OPENBLAS_FC_VER="$(basename "$(dirname "$(dirname "$OPENBLAS_FC")")")"
    OPENBLAS_ONEAPI_ROOT="$(cd "$(dirname "$(dirname "$OPENBLAS_FC")")/../.." && pwd -P)"
    OPENBLAS_RUNTIME_LIB="$OPENBLAS_ONEAPI_ROOT/$OPENBLAS_FC_VER/lib"
    OPENBLAS_LIBS="$OPENBLAS_FC_LIB"
    [[ -d "$OPENBLAS_RUNTIME_LIB" ]] && OPENBLAS_LIBS="$OPENBLAS_LIBS:$OPENBLAS_RUNTIME_LIB"
    export LD_LIBRARY_PATH="$OPENBLAS_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    # The build needs that path to run OpenBLAS's own test programs, and the *harness* needs it
    # again at run time: an `ifx`-built OpenBLAS leaves Fortran-runtime references in the archive
    # it produces, so a binary linked against it will not start without `libifcoremt`/`libsvml` on
    # the loader path. An arm whose binary cannot be executed directly is not an arm this campaign
    # can record -- the recorder executes the harness itself, with the ambient thread variables
    # removed, not through a shell that arranges a search path. So the paths are burnt into the
    # binary as an rpath instead, and recorded with the compiler they belong to. `--disable-new-dtags`
    # matters: a `RUNPATH` is not consulted for a *transitive* dependency, and the runtime pieces
    # are exactly that (`libifcoremt` needs `libsvml`, which lives in the second directory), so the
    # link has to emit an inherited `RPATH` rather than a `RUNPATH`.
    OPENBLAS_RPATH="$(printf -- '-C link-arg=-Wl,--disable-new-dtags -C link-arg=-Wl,-rpath,%s ' $OPENBLAS_FC_LIB $OPENBLAS_RUNTIME_LIB)"
    export RUSTFLAGS="${RUSTFLAGS:-} ${OPENBLAS_RPATH% }"
    VENDOR_BUILD_ENV="OPENBLAS_DYNAMIC_ARCH=1 FC=$OPENBLAS_FC LD_LIBRARY_PATH=$OPENBLAS_LIBS rpath=$OPENBLAS_FC_LIB${OPENBLAS_RUNTIME_LIB:+:$OPENBLAS_RUNTIME_LIB}"
    ;;
  mkl)
    FEATURES="fbench/link-mkl"
    # The installed MKL, named by versioned path. The threading runtime is linked eagerly because
    # MKL's Intel OpenMP layer is loaded at run time with an undefined `omp_get_num_procs`, and
    # `--as-needed` would otherwise drop the library nothing references.
    MKLROOT="$(newest_dir "${MKLROOT:-/opt/intel/oneapi/mkl/*/}")"
    MKL_THREADING_LIB="${MKL_THREADING_LIB:-$(newest_dir "/opt/intel/oneapi/compiler/*/lib/libiomp5.so")}"
    [[ -n "$MKLROOT" && -n "$MKL_THREADING_LIB" ]] || {
      echo "ERROR: no installed MKL found; set MKLROOT and MKL_THREADING_LIB" >&2; exit 1; }
    # What is recorded has to name the tree that was used, not a symlink that can be moved under it:
    # `MKLROOT=.../latest` in a manifest records nothing a reader can act on.
    MKLROOT="$(cd "$MKLROOT" && pwd -P)"
    MKL_THREADING_LIB="$(readlink -f "$MKL_THREADING_LIB")"
    export MKLROOT MKL_THREADING_LIB
    VENDOR_BUILD_ENV="MKLROOT=$MKLROOT MKL_THREADING_LIB=$MKL_THREADING_LIB"
    ;;
  *)
    echo "ERROR: unknown --vendor '$VENDOR'; this harness knows openblas and mkl" >&2; exit 1 ;;
esac

# ------------------------------------------------------------ fork API generation
# The fork is measured at two parallelism API generations without editing the library: the 0.24.x
# line (`Par` is `Copy`, workspace queries take a `Par`, a process-wide setting exists) and the 0.25
# line (`Par<'team>`, `ParShape` for workspace queries, no process-wide setting). The harness's fork
# rows call explicit-parallelism entry points only (`backend::fork_par`); the one spelling that
# differs is the type a `*_scratch` function takes, and the harness feature `fork-par-shape` picks it.
# The generation is read from the checkout itself (`faer/src/lib.rs` declares `pub enum ParShape`
# or it does not), so it cannot disagree with the revision that is linked.
if grep -q '^pub enum ParShape' "$FORK_DIR/faer/src/lib.rs"; then
    FORK_API="parshape"
    FEATURES="$FEATURES,fbench/fork-par-shape"
else
    FORK_API="par"
fi
# The parallelism API generation: from tensor4all/faer#85 on the only route is a threadteam team the
# caller owns (`faer::team::with_team`), and the fork's `Par` cannot be built from a budget (it is
# exclusive authority, not `Copy`). Read from the checkout like the generation above, so the harness
# cannot disagree with the revision it links; a checkout without `with_team` keeps the old path.
if grep -q 'pub fn with_team' "$FORK_DIR/faer/src/utils/team.rs"; then
    TEAM_API="threadteam"
    FEATURES="$FEATURES,fbench/fork-team"
else
    TEAM_API="rayon"
fi
# ------------------------------------------------------------------------------

# --------------------------------------------------------------- path injection
# The measured libraries are the checkouts the caller named, not registry downloads, and Cargo has
# no `--path` for a dependency. So the registry entries the harness declares are patched to those
# checkout paths here. `--config` keeps the tracked manifests untouched: a checkout path belongs to
# the measuring host and must not be committed. The crate directory inside each checkout is `faer/`
# -- the workspace member that carries the package whose lib is named `faer`, published by the fork
# as `t4a-faer` and by upstream as `faer` -- so the two patches cannot collide. If the harness
# declares its dependencies another way (a `[patch]` section of its own, or a generated manifest),
# this is the one place that has to change; the build below is the same either way.
PATCH_ARGS=(
    --config "patch.crates-io.t4a-faer.path=\"$FORK_DIR/faer\""
    --config "patch.crates-io.faer.path=\"$UPSTREAM_DIR/faer\""
)
# ------------------------------------------------------------------------------

# --------------------------------------------------------- patch resolution check
# A `[patch]` is silently ignored when the locked version is not the version the checkout provides:
# cargo warns and links the registry crate instead, so the build would record a revision it did not
# use. Resolve the graph first and refuse unless both measured crates come from the named checkouts.
# The committed `Cargo.lock` is the harness's own resolution; a path-patched fork whose version or
# dependency set differs from the one the lock names (0.24.x against 0.25.0 are different lock
# entries) makes cargo rewrite it. That rewrite belongs to the build of *that checkout*, not to the
# harness source: the recorder's clean-tree rule would otherwise refuse the run after the build. So
# the lock the build resolved is kept beside the binary (`Cargo.lock.resolved`, hashed into
# `BUILD_LOCK_SHA256`) and the committed lock is put back byte for byte. Read the resolved lock's
# diff against the committed one unfiltered (`diff Cargo.lock <target>/Cargo.lock.resolved`): the
# only acceptable differences are the fork's own version and dependency lines.
lock_backup="$(mktemp)"
cp "$PROJECT_DIR/Cargo.lock" "$lock_backup"
restore_lock() { cp "$lock_backup" "$PROJECT_DIR/Cargo.lock"; rm -f "$lock_backup"; }
trap restore_lock EXIT

bash "$(dirname "${BASH_SOURCE[0]}")/faer_patch_preflight.sh" \
    --manifest "$PROJECT_DIR/Cargo.toml" \
    --fork-manifest "$FORK_DIR/faer/Cargo.toml" \
    --upstream-manifest "$UPSTREAM_DIR/faer/Cargo.toml" \
    "${PATCH_ARGS[@]}" || exit 1

bin_args=()
for bin in "${BINS[@]}"; do bin_args+=(--bin "$bin"); done


RUSTC_WRAPPER= CARGO_TARGET_DIR="$target" \
    env -u OPENBLAS_NUM_THREADS -u OMP_NUM_THREADS -u RAYON_NUM_THREADS -u MKL_NUM_THREADS \
        -u MKL_DOMAIN_NUM_THREADS -u MKL_INTERFACE_LAYER -u MKL_THREADING_LAYER -u MKL_DYNAMIC \
        OPENBLAS_DYNAMIC_ARCH=1 \
    cargo build --manifest-path "$PROJECT_DIR/Cargo.toml" "${profile_flag[@]}" \
    -p fbench --features "$FEATURES" "${PATCH_ARGS[@]}" "${bin_args[@]}" >&2

bin_dir="$target/$subdir"
cp "$PROJECT_DIR/Cargo.lock" "$target/Cargo.lock.resolved"
lock_sha="$(sha256sum "$target/Cargo.lock.resolved" | cut -d' ' -f1)"
{
    printf 'FAER_FORK_REV=%q\n' "$fork_rev"
    printf 'FAER_FORK_DIRTY=%q\n' "$fork_dirty"
    printf 'FAER_FORK_DIR=%q\n' "$FORK_DIR"
    printf 'FAER_FORK_VERSION=%q\n' "$fork_version"
    printf 'FAER_FORK_FEATURES=%q\n' "$(features_of faer_fork)"
    printf 'FAER_UPSTREAM_REV=%q\n' "$up_rev"
    printf 'FAER_UPSTREAM_DIRTY=%q\n' "$up_dirty"
    printf 'FAER_UPSTREAM_DIR=%q\n' "$UPSTREAM_DIR"
    printf 'FAER_UPSTREAM_VERSION=%q\n' "$up_version"
    printf 'FAER_UPSTREAM_FEATURES=%q\n' "$(features_of faer_upstream)"
    printf 'FAER_UPSTREAM_REF=%q\n' "$up_ref"
    printf 'BIN_DIR=%q\n' "$bin_dir"
    printf 'BUILD_FEATURES=%q\n' "$FEATURES"
    printf 'BUILD_ENV=%q\n' "$VENDOR_BUILD_ENV"
    printf 'BUILD_VENDOR=%q\n' "$VENDOR"
    printf 'BUILD_FORK_API=%q\n' "$FORK_API"
    printf 'BUILD_TEAM_API=%q\n' "$TEAM_API"
    printf 'BUILD_LOCK_SHA256=%q\n' "$lock_sha"
} > "$OUT_FILE"
echo "built ${BINS[*]} for fork $fork_rev and upstream $up_rev (dirty=$fork_dirty/$up_dirty) in $bin_dir" >&2
