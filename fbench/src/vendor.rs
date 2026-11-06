//! The vendor BLAS/LAPACK library this binary was built against.
//!
//! One binary links one vendor library, so the identity below names the arm and
//! `vendor.name` is checked against the `--vendor` the recorder asked for. A row
//! labelled `1T` is only a single-thread measurement if the vendor library also
//! ran on one thread; LAPACK and BLAS own their threading, so the harness sets
//! the vendor's budget and reads it back rather than assuming it.
//!
//! `vendor.config`, `vendor.corename`, `vendor.parallel` and `vendor.procs` are
//! printed only when the linked library actually reports them. MKL reports none
//! of them (it names no dispatched kernel and no processor count), so its identity
//! has just the name, the linkage and the version string; leaving a field out is
//! how the manifest distinguishes "not reported" from a value nobody read back.

/// The vendor library's name: `openblas`, `mkl`, `aocl`, or `none`.
pub const NAME: &str = if cfg!(feature = "link-mkl") {
    "mkl"
} else if cfg!(feature = "link-openblas") {
    "openblas"
} else if cfg!(feature = "link-aocl") {
    "aocl"
} else {
    "none"
};

/// How the vendor library was linked.
pub fn linkage() -> &'static str {
    if cfg!(any(feature = "link-mkl", feature = "link-aocl")) {
        "shared"
    } else if cfg!(feature = "link-openblas") {
        "static"
    } else {
        "none"
    }
}

/// What the linked vendor library says about itself.
#[derive(Debug, Clone)]
pub struct Identity {
    pub name: &'static str,
    pub linkage: &'static str,
    pub version: Option<String>,
    pub config: Option<String>,
    pub corename: Option<String>,
    pub parallel: Option<String>,
    pub procs: Option<usize>,
}

impl Identity {
    /// The identity as `key=value` lines, for a manifest to record verbatim.
    pub fn lines(&self) -> Vec<(String, String)> {
        let mut out = vec![
            ("vendor.name".to_owned(), self.name.to_owned()),
            ("vendor.linkage".to_owned(), self.linkage.to_owned()),
        ];
        for (key, value) in [
            ("vendor.version", &self.version),
            ("vendor.config", &self.config),
            ("vendor.corename", &self.corename),
            ("vendor.parallel", &self.parallel),
        ] {
            if let Some(value) = value {
                out.push((key.to_owned(), value.clone()));
            }
        }
        if let Some(procs) = self.procs {
            out.push(("vendor.procs".to_owned(), procs.to_string()));
        }
        out
    }
}

/// The linked vendor library's identity.
pub fn identity() -> Identity {
    Identity {
        name: NAME,
        linkage: linkage(),
        version: sys::version(),
        config: sys::config(),
        corename: sys::corename(),
        parallel: sys::parallel(),
        procs: sys::procs(),
    }
}

/// Set the vendor library's thread budget and read it back.
pub fn set_threads(threads: usize) -> Result<(), String> {
    if NAME == "none" {
        return if threads == 1 {
            Ok(())
        } else {
            Err("no vendor library is linked, so a vendor thread budget cannot be set".to_owned())
        };
    }
    sys::set_threads(threads);
    // LAPACKE scans every input matrix for NaN before the call unless told not to; faer does no
    // such pass, so the vendor row turns it off once, before anything is measured, and checks it.
    sys::set_nancheck(false);
    if sys::nancheck() != Some(false) {
        return Err("LAPACKE_set_nancheck(0) did not take effect".to_owned());
    }
    require_threads(threads)
}

/// Whether LAPACKE validates inputs for NaN (`None`: no vendor library is linked).
pub fn nancheck() -> Option<bool> {
    if NAME == "none" { None } else { sys::nancheck() }
}

/// The library-reported maximum thread budget, not an active-worker trace.
pub fn reported_threads() -> Option<usize> {
    sys::threads()
}

/// Check that the vendor library is configured for `threads` threads.
pub fn require_threads(threads: usize) -> Result<(), String> {
    if NAME == "none" {
        return Ok(());
    }
    match sys::threads() {
        Some(actual) if actual == threads => Ok(()),
        Some(actual) => Err(format!(
            "{NAME} reports {actual} threads, but this row declares {threads}"
        )),
        None => Err(format!("{NAME} does not report its thread count")),
    }
}

#[cfg(feature = "link-mkl")]
mod sys {
    use core::ffi::CStr;
    use core::ffi::{c_char, c_int};

    unsafe extern "C" {
        fn mkl_get_version_string(buffer: *mut c_char, length: c_int);
        // `mkl_set_num_threads` segfaults in this MKL 2026.1 install before any
        // computational call (reproduced with a plain C program linked the same way),
        // so the budget is set through the Intel OpenMP layer MKL is linked against;
        // `mkl_get_max_threads` reads it back either way.
        fn omp_set_num_threads(threads: c_int);
        fn mkl_get_max_threads() -> c_int;
    }

    /// The reported version, e.g. `2026.1`, read out of MKL's own version string.
    pub fn version() -> Option<String> {
        let mut buffer = [0 as c_char; 256];
        // SAFETY: MKL writes at most `length` bytes into a buffer this call owns.
        unsafe { mkl_get_version_string(buffer.as_mut_ptr(), buffer.len() as c_int) };
        // SAFETY: the buffer is NUL-terminated on return and stays alive here.
        let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let rest = text.split("Version ").nth(1)?;
        let version: String = rest
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '.')
            .collect();
        (!version.is_empty()).then_some(version)
    }

    pub fn set_threads(threads: usize) {
        // SAFETY: a plain setter on the Intel OpenMP runtime's thread count, which
        // is the layer this MKL build uses.
        unsafe { omp_set_num_threads(threads as c_int) }
    }

    unsafe extern "C" {
        fn LAPACKE_set_nancheck(flag: c_int);
        fn LAPACKE_get_nancheck() -> c_int;
    }

    pub fn set_nancheck(on: bool) {
        // SAFETY: a process-wide LAPACKE flag, set once before any measured call.
        unsafe { LAPACKE_set_nancheck(on as c_int) }
    }

    pub fn nancheck() -> Option<bool> {
        // SAFETY: reads the flag set above.
        Some(unsafe { LAPACKE_get_nancheck() } != 0)
    }

    pub fn threads() -> Option<usize> {
        // SAFETY: no arguments beyond reading MKL's current thread count.
        let threads = unsafe { mkl_get_max_threads() };
        (threads > 0).then_some(threads as usize)
    }

    pub fn config() -> Option<String> {
        None
    }
    pub fn corename() -> Option<String> {
        None
    }
    pub fn parallel() -> Option<String> {
        None
    }
    pub fn procs() -> Option<usize> {
        None
    }
}

#[cfg(feature = "link-openblas")]
mod sys {
    use core::ffi::{c_char, c_int, CStr};

    unsafe extern "C" {
        fn openblas_set_num_threads(num_threads: c_int);
        fn openblas_get_num_threads() -> c_int;
        fn openblas_get_num_procs() -> c_int;
        fn openblas_get_config() -> *const c_char;
        fn openblas_get_corename() -> *const c_char;
        fn openblas_get_parallel() -> c_int;
    }

    /// A `char *` the vendor library owns and keeps alive.
    fn string_of(pointer: *const c_char) -> Option<String> {
        if pointer.is_null() {
            return None;
        }
        // SAFETY: the vendor returns a NUL-terminated string valid for the process.
        Some(
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned(),
        )
    }

    /// `OpenBLAS 0.3.32 NO_AFFINITY COOPERLAKE` -> `0.3.32`.
    pub fn version() -> Option<String> {
        let config = config()?;
        let mut fields = config.split_whitespace();
        match (fields.next(), fields.next()) {
            (Some("OpenBLAS"), Some(version)) => Some(version.to_owned()),
            _ => None,
        }
    }

    pub fn config() -> Option<String> {
        // SAFETY: returns a pointer to a static string owned by the vendor.
        string_of(unsafe { openblas_get_config() })
    }

    pub fn corename() -> Option<String> {
        // SAFETY: as `config`.
        string_of(unsafe { openblas_get_corename() })
    }

    pub fn parallel() -> Option<String> {
        // SAFETY: no arguments, returns 0 sequential / 1 pthread / 2 openmp.
        match unsafe { openblas_get_parallel() } {
            0 => Some("sequential".to_owned()),
            1 => Some("pthread".to_owned()),
            2 => Some("openmp".to_owned()),
            _ => None,
        }
    }

    pub fn set_threads(threads: usize) {
        // SAFETY: a plain setter on the vendor's global thread count.
        unsafe { openblas_set_num_threads(threads as c_int) }
    }

    unsafe extern "C" {
        fn LAPACKE_set_nancheck(flag: c_int);
        fn LAPACKE_get_nancheck() -> c_int;
    }

    pub fn set_nancheck(on: bool) {
        // SAFETY: a process-wide LAPACKE flag, set once before any measured call.
        unsafe { LAPACKE_set_nancheck(on as c_int) }
    }

    pub fn nancheck() -> Option<bool> {
        // SAFETY: reads the flag set above.
        Some(unsafe { LAPACKE_get_nancheck() } != 0)
    }

    pub fn threads() -> Option<usize> {
        // SAFETY: no arguments beyond reading the vendor's thread count.
        let threads = unsafe { openblas_get_num_threads() };
        (threads > 0).then_some(threads as usize)
    }

    pub fn procs() -> Option<usize> {
        // SAFETY: as `threads`.
        let procs = unsafe { openblas_get_num_procs() };
        (procs > 0).then_some(procs as usize)
    }
}

#[cfg(feature = "link-aocl")]
mod sys {
    //! AMD AOCL 5.x: LAPACKE from `libflame` (AOCL-LAPACK), CBLAS from the threaded BLIS
    //! (`libblis-mt`, AOCL-BLAS), both on GNU OpenMP.
    //!
    //! Three thread counts govern one call, and all three are thread-local to the caller:
    //! OpenMP's `omp_get_max_threads()`, BLIS's own count (its `BLIS_NUM_THREADS`), and
    //! libflame's (its `FLA_NUM_THREADS`, `FLA_Context.c` `fla_thread_update_rntm_from_env`).
    //! With neither library variable set, both libraries take the OpenMP value, so the budget is
    //! set the way the MKL arm sets it (`omp_set_num_threads`, the standard knob) and all three
    //! are read back; a stray `BLIS_NUM_THREADS`/`FLA_NUM_THREADS` overrides the OpenMP value for
    //! its library and is caught by the read-back instead of silently changing the row. The
    //! harness sets and reads them on the thread that times the vendor row (`main` calls
    //! `configure_threads` before any row, and `measure_row` times vendor rows on that thread).
    //! libflame's `dgetrf` also uses its thread count to choose an algorithm, not only a width
    //! (`FLA_getrf.c`: the AVX-512 small path is taken only when it is not 1).
    use core::ffi::{c_char, c_int, c_long, CStr};

    unsafe extern "C" {
        fn omp_set_num_threads(threads: c_int);
        fn omp_get_max_threads() -> c_int;
        fn bli_thread_get_num_threads() -> c_long;
        fn bli_info_get_version_str() -> *const c_char;
        fn bli_arch_query_id() -> c_int;
        fn bli_arch_string(id: c_int) -> *const c_char;
        // Exported by libflame but not in its public header: the per-thread count it will use.
        fn fla_thread_get_num_threads() -> c_int;
        fn fla_arch_query_id() -> c_int;
        fn aocl_fla_init();
        fn FLA_Get_AOCL_Version() -> *const c_char;
        fn LAPACKE_set_nancheck(flag: c_int);
        fn LAPACKE_get_nancheck() -> c_int;
    }

    fn string_of(pointer: *const c_char) -> Option<String> {
        if pointer.is_null() {
            return None;
        }
        // SAFETY: both libraries return NUL-terminated strings they own for the process.
        let text = unsafe { CStr::from_ptr(pointer) }.to_string_lossy().into_owned();
        Some(text.split_whitespace().collect::<Vec<_>>().join(" "))
    }

    /// `AOCL-BLAS 5.2.0 Build 20251223` -> `5.2.0`; libflame's own string is in `config`.
    pub fn version() -> Option<String> {
        // SAFETY: no arguments; returns a static string.
        let blis = string_of(unsafe { bli_info_get_version_str() })?;
        blis.split_whitespace().nth(1).map(str::to_owned)
    }

    /// Both libraries' own version strings and the files the loader actually mapped for them,
    /// because the loader path is set by the caller (`LD_LIBRARY_PATH`), not by the binary.
    pub fn config() -> Option<String> {
        // SAFETY: as `version`.
        let blis = string_of(unsafe { bli_info_get_version_str() })?;
        // SAFETY: as `version`.
        let flame = string_of(unsafe { FLA_Get_AOCL_Version() })?;
        let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
        let mut mapped: Vec<&str> = maps
            .lines()
            .filter_map(|l| l.split_whitespace().nth(5))
            .filter(|p| ["libflame", "libblis", "libaoclutils", "libgomp"].iter().any(|k| p.contains(k)))
            .collect();
        mapped.sort_unstable();
        mapped.dedup();
        Some(format!("{blis}; {flame}; mapped={}", mapped.join(":")))
    }

    /// The kernel family each library dispatched to on this CPU: BLIS's configuration name
    /// (`zen5`) and libflame's ISA level (`fla_arch_t`: 3 = AVX2, 4 = AVX-512).
    pub fn corename() -> Option<String> {
        // SAFETY: the context is initialised once; the queries read it.
        let (blis, flame) = unsafe {
            aocl_fla_init();
            (string_of(bli_arch_string(bli_arch_query_id())), fla_arch_query_id())
        };
        let flame = match flame {
            3 => "avx2".to_owned(),
            4 => "avx512".to_owned(),
            other => format!("id{other}"),
        };
        Some(format!("blis={}, flame={flame}", blis?))
    }

    pub fn parallel() -> Option<String> {
        Some("openmp".to_owned())
    }

    pub fn procs() -> Option<usize> {
        None
    }

    pub fn set_threads(threads: usize) {
        // SAFETY: sets the calling thread's OpenMP ICV, which both libraries read per call.
        unsafe { omp_set_num_threads(threads as c_int) }
    }

    pub fn set_nancheck(on: bool) {
        // SAFETY: a process-wide LAPACKE flag, set once before any measured call.
        unsafe { LAPACKE_set_nancheck(on as c_int) }
    }

    pub fn nancheck() -> Option<bool> {
        // SAFETY: reads the flag set above.
        Some(unsafe { LAPACKE_get_nancheck() } != 0)
    }

    /// The budget only when OpenMP, BLIS and libflame agree on it; otherwise the disagreeing
    /// value, so `require_threads` names a mismatch instead of passing on one of the three.
    pub fn threads() -> Option<usize> {
        // SAFETY: no arguments; each reads the calling thread's count.
        let (omp, blis, flame) = unsafe {
            (omp_get_max_threads() as i64, bli_thread_get_num_threads() as i64, fla_thread_get_num_threads() as i64)
        };
        let value = if omp == blis && blis == flame { omp } else if blis != omp { blis } else { flame };
        (value > 0).then_some(value as usize)
    }
}

#[cfg(not(any(feature = "link-mkl", feature = "link-openblas", feature = "link-aocl")))]
mod sys {
    pub fn version() -> Option<String> {
        None
    }
    pub fn config() -> Option<String> {
        None
    }
    pub fn corename() -> Option<String> {
        None
    }
    pub fn parallel() -> Option<String> {
        None
    }
    pub fn procs() -> Option<usize> {
        None
    }
    pub fn set_threads(_threads: usize) {}
    pub fn threads() -> Option<usize> {
        None
    }
    pub fn set_nancheck(_on: bool) {}
    pub fn nancheck() -> Option<bool> {
        None
    }
}
