//! The atomics the lock-free [`PackedTable`](super::PackedTable) is built on: the core ones,
//! or loom's under `cfg(loom)`, so the loom tests can explore every interleaving of them.
//!
//! loom is a dev-dependency, so `--cfg loom` builds only the tests (`task test:loom`).

#[cfg(loom)]
pub(crate) use loom::sync::atomic::{AtomicU64, Ordering};

#[cfg(not(loom))]
pub(crate) use core::sync::atomic::{AtomicU64, Ordering};
