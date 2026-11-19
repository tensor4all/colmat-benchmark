//! Make the MKL arm link and run the way the campaign requires.
//!
//! Two things `intel-mkl-src` does not do, both of which matter because the
//! recorder executes the harness binary directly with no loader search path to
//! arrange and no environment of its own:
//!
//! * it emits `-L`/`-l` for the dynamic MKL interface, threading layer and core
//!   plus the Intel OpenMP runtime, but no rpath, so the versioned directories
//!   `scripts/build_for_faer_rev.sh` resolved have to be recorded here;
//! * the linker's `--as-needed` default drops `mkl_intel_thread`, `mkl_core` and
//!   `iomp5`, because the harness's own undefined symbols (`cblas_?gemm`,
//!   `LAPACKE_*`) are all defined by `mkl_intel_lp64`. The interface layer then
//!   fails to relocate its own `mkl_blas_*` references at load time. The three
//!   are forced back in after `--no-as-needed`.
//!
//! Nothing here follows a moving `latest` symlink: `MKLROOT` and
//! `MKL_THREADING_LIB` are the versioned paths the build script exports, and the
//! binary records what was linked. The OpenBLAS arm is linked statically
//! (`openblas-src/static`), so it needs neither of these and this does nothing.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=MKL_THREADING_LIB");
    println!("cargo:rerun-if-env-changed=AOCL_ROOT");
    let arms = ["CARGO_FEATURE_LINK_MKL", "CARGO_FEATURE_LINK_OPENBLAS", "CARGO_FEATURE_LINK_AOCL"]
        .iter()
        .filter(|f| std::env::var_os(f).is_some())
        .count();
    if arms > 1 {
        panic!("one binary links one vendor library: enable exactly one of link-mkl, link-openblas, link-aocl");
    }
    if std::env::var_os("CARGO_FEATURE_LINK_AOCL").is_some() {
        link_aocl();
        return;
    }
    if std::env::var_os("CARGO_FEATURE_LINK_MKL").is_none() {
        return;
    }
    let mut dirs = Vec::new();
    if let Some(root) = std::env::var_os("MKLROOT") {
        dirs.push(PathBuf::from(root).join("lib"));
    }
    if let Some(threading) = std::env::var_os("MKL_THREADING_LIB") {
        if let Some(parent) = PathBuf::from(threading).parent() {
            dirs.push(parent.to_path_buf());
        }
    }
    for dir in &dirs {
        if dir.is_dir() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
        }
    }
    // The `link-mkl` feature selects LP64 + Intel OpenMP, so the chain is fixed.
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    for lib in ["mkl_intel_lp64", "mkl_intel_thread", "mkl_core", "iomp5"] {
        println!("cargo:rustc-link-arg=-l{lib}");
    }
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
}

/// The AOCL arm: LAPACKE from `libflame`, CBLAS and the BLAS libflame calls from the threaded
/// BLIS (`libblis-mt`), and `libaoclutils`, which libflame needs for its CPU-feature query
/// (`au_cpuid_has_flags`, how it picks its AVX2/AVX-512 paths). `AOCL_ROOT` is the versioned
/// install tree the build script resolves: `<prefix>/<ver>/gcc` for AOCL 5.x,
/// `<prefix>/<ver>/gcc/MT` for AOCL 6.x (which installs separate single- and multi-threaded
/// trees). The LP64 libraries are named by their own directory, not through the installer's
/// `lib -> lib_LP64` convenience link.
///
/// The recorder executes the harness directly, with no loader search path of its own, so the
/// directory is recorded in the binary, as for MKL. It is written as an inherited `DT_RPATH`
/// (`--disable-new-dtags`), not a `DT_RUNPATH`: a `RUNPATH` is not consulted for libflame's own
/// `NEEDED libaoclutils.so` / `libblis-mt.so.N`, an `RPATH` is. AOCL stays out of `ldconfig`, and
/// the harness reports the files the loader actually mapped (`vendor.config`), so a run that
/// loaded any other copy is visible in its identity.
fn link_aocl() {
    let root = std::env::var_os("AOCL_ROOT")
        .map(PathBuf::from)
        .expect("link-aocl needs AOCL_ROOT (the versioned AOCL tree, e.g. /opt/AMD/aocl/6.0.0/gcc/MT)");
    let lib = root.join("lib_LP64");
    for name in ["libflame.so", "libblis-mt.so", "libaoclutils.so"] {
        assert!(lib.join(name).exists(), "no {name} in {}", lib.display());
    }
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    for name in ["flame", "blis-mt", "aoclutils", "m"] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    // The harness sets and reads the OpenMP budget itself (`vendor.rs`), so it names the GNU
    // OpenMP runtime both AOCL libraries already need, by the same soname, rather than relying on
    // the linker to find `omp_*` through their dependencies (lld does not).
    println!("cargo:rustc-link-arg=-l:libgomp.so.1");
}
