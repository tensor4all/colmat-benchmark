# Measure the renamed colmat package

## Decisions

The harness depends directly on package and library `colmat` from 0.34 onward,
patches that package to the requested checkout, and verifies its resolved manifest
before building. Upstream remains a separate `faer` dependency. Historical pin
paths and CSV engine names remain stable so prior recordings remain readable.
The lock update follows colmat's actual dependencies, including threadteam 0.6
and t4a-gemm; it does not update unrelated package versions.

The build script accepts `--vendor none` for library-only diagnostics on hosts
without the vendor build toolchain. This option is not a new public recorder arm
and links no vendor. Formal recording gates remain in force.

## Verification conclusions and constraints

The release harness builds against colmat 0.34.0 at the current local checkout
and upstream faer 0.24.4. All 12 Rust harness tests and 47 Python tests pass,
including the rejection of a legacy package in place of colmat and the test that
checks operations use the lent team token. Vendor-linked builds are not checked
on this host. This change makes no performance comparison or correctness claim
about an unmeasured upstream revision.
