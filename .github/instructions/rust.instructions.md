---
applyTo: "**/*.rs"
---

# Rust code

- Every module (`//!`), type, field, variant, function, constant and macro has a doc
  comment, private and `pub(crate)` items included. Docs state intent, invariants, units and
  error/panic conditions (`# Errors`, `# Panics`, `# Safety`). They do not restate the name.
- No `mod.rs`: modules with children are `foo.rs` plus `foo/`. Integration tests live under
  `tests/it/`.
- No separator comments (`// -----`, `// =====`). If a file needs sections, split it into
  submodules.
- Leave a blank line after the closing `}` of a block statement when another statement
  follows, and between logical phases of a function. No lumped code.
- One level of abstraction per function. Long loops (simulations, game ticks, solvers) are
  short drivers that call documented step functions. If a block needs a "what" comment,
  extract it into a function named after that comment.
- `clippy::pedantic` clean. Suppression only with a scoped
  `#[expect(lint, reason = "...")]`. Never `#[allow]`, and never crate-level suppression.
- No `unsafe` unless the crate opts out of `#![forbid(unsafe_code)]` with a documented
  reason. Each unsafe block has one operation and a `// SAFETY:` comment.
- The crate is `#![no_std]` with `alloc`, in every feature combination except `parallel`,
  which links `std` for its threads (the CI `no-std` job builds the rest of the powerset). Its runtime dependencies are search-core and libm, plus
  hashbrown and rand_core behind features. Flag any new `[dependencies]` entry, any `std`
  use, and any feature-gated item without its `#[cfg]` on the re-export in `lib.rs`.
- Scores returned by `Game::evaluate` are from `Side::Max`'s point of view, whoever is to
  move, and the side to move belongs to the position, so one side may move several times in
  a row. A search that changes perspective negates only where the side to move changes,
  never once per ply. Every search must agree with `minimax` in the cross-checks under
  `tests/it/`.
- Public API changes go with a matching update to the README scope list and usage example
  and the crate docs in `lib.rs`. Name every public API change in the PR.
- Mermaid diagrams in doc comments must match the code they describe. Update them when the
  structure changes.
