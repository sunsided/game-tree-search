# Incremental evaluation and make/unmake

Design note for [#51](https://github.com/sunsided/game-tree-search/issues/51): how the crate
should support NNUE-style incremental evaluation, and whether alpha-beta needs make/unmake
for it.

## Decision in short

- Incremental evaluation already works with the current API. A game keeps its accumulator
  in the position and updates it by a delta in `Game::play` (copy-make). On the measured
  toy network, that is 3 to 5 times cheaper than recomputing the accumulator in `evaluate`.
- Make/unmake would save only the copy, not the update. On the same network, an in-place
  accumulator stack ran 3 to 4% fewer instructions than copy-make, and the wall-clock times
  were within noise. That does not justify a second search path through every
  alpha-beta-family search.
- So: document the copy-make accumulator pattern now, and keep the `Incremental` design
  below on file. Build it when a game crosses the trigger in
  [When to build make/unmake](#when-to-build-makeunmake).

## Two separate questions

Issue #51 mixes two things.

1. **Incremental evaluation.** The evaluator updates a cached intermediate result (the NNUE
   accumulator) by what a move changed, instead of recomputing it from the whole position.
   This is about where the cache lives and when it is updated.
2. **Make/unmake.** The search mutates one position in place along the current line and
   undoes each move afterwards, instead of building a new position per move. This is about
   how the search holds positions.

Engines often combine them, but the first does not need the second. The rest of this note
treats them apart.

## What engines do

From the nnue-pytorch documentation (`docs/nnue.md`), quoted:

- The accumulator is the first layer's output, kept with the position: "Instead of
  recomputing the first set of hidden neurons for each position we can keep them as part of
  the position's state, and update it on each move based on what features (columns) were
  added or removed!" (Basics, Accumulator)
- A move adds or subtracts weight columns: "the feature `i` was removed from the input (1 ->
  0) - subtract column `i` of the weight matrix from the accumulator", and the reverse for an
  added feature. (Basics, Accumulator)
- Some moves need a full recomputation: for a king move in HalfKP, "All features are
  changed, so an accumulator refresh is executed." (Basics, HalfKP)
- Storage: "The accumulator can be represented by an array that is stored along other
  position state information on the search stack." (Forward pass implementation,
  Accumulator)
- Undo does not touch it: "it is best implemented such that the accumulator is not updated
  when undoing a move. Instead, it is simply stored on the search stack". (Basics,
  Accumulator)
- Timing: "The accumulator can either be updated lazily on evaluation, or on each move."
  and "Whether it's better to update lazily or eagerly depends on the number of evaluations
  done during search." (Forward pass implementation, Accumulator)
- Size: `int16_t v[2][N]`, one row per perspective, where Stockfish moved from N = 256 to
  N = 1024. (Quantization; Consideration of networks size and cost) That is 1 KB to 4 KB
  per accumulator.

So even an engine built on make/unmake keeps one accumulator per ply and computes the
child's from the parent's. That is copy-make for the accumulator. The board itself is what
make/unmake saves copying, and a board is usually far smaller than 1 to 4 KB.

## Measurements

The probe used the test suite's `RandomTree`: up to 8 moves per position, 4096 positions
per ply, iterative deepening to depth 9 with a `SlotTable`. Every variant searched the same
15,193 positions. The toy network had 64 features, 32 active features per position, and
an `int16` accumulator of `2 × N` lanes. Each move removed one feature and added one, in
both perspectives, and `evaluate` computed a clipped dot product of the accumulator. The
search's own score came from the tree, so the search was identical across variants.

Variants:

- **none:** no accumulator, the baseline.
- **refresh:** `evaluate` rebuilds the accumulator from the 32 active features.
- **copy + delta:** the position owns the accumulator; `play` clones it and applies the
  move's delta, and `evaluate` reads it.
- **stack:** the game keeps one accumulator per ply in a `RefCell<Vec<_>>`. `play` writes
  the child's slot from the parent's slot plus the delta, and the position only carries its
  ply. This is what make/unmake would allow, without copying the rest of the position.

Instructions under callgrind (deterministic):

| variant | N = 256 (1 KB) | N = 1024 (4 KB) |
|---|---|---|
| none | 6.7M | 6.7M |
| refresh | 100.1M | 481.2M |
| copy + delta | 34.5M | 97.0M |
| stack | 33.4M (−3%) | 92.9M (−4%) |

Median wall-clock time of 31 interleaved runs:

| variant | N = 256 | N = 1024 |
|---|---|---|
| none | 1.04 ms | 1.04 ms |
| refresh | 7.92 ms | 31.9 ms |
| copy + delta | 2.86 ms | 8.80 ms |
| stack | 2.98 ms | 8.30 ms |

The copy + delta variant even allocated a fresh `Vec` per move and still came within a few
percent of the stack. The delta, a read-modify-write of every lane from two weight columns,
dominates; copying the parent's lanes on the way costs little more. A plain copy-make probe
without an accumulator agreed: positions of up to about 500 bytes cost nothing measurable
to copy on this tree.

Limits: one machine, a toy network without SIMD, and a tree whose positions cost almost
nothing besides the accumulator. A real engine's board, move generation and evaluation make
the copy an even smaller share.

## Recommendation now

1. **Document the copy-make accumulator pattern** in the `Game` docs, next to the
   costly-positions section from #47. Keep the accumulator in the position and update it in
   `play` by the move's delta; recompute it fully where the network needs that, such as a
   king move. This is a docs change with no new API.
2. **Mention the eager-update cost.** A searcher calls `play` before it probes the
   transposition table for the child, so a table hit still pays the child's delta. Table
   hits were about a quarter of the positions in the Lazy SMP measurements of #48. See
   [Probing before play](#probing-before-play) for the option that would avoid it.
3. **Do not build make/unmake yet.**

## When to build make/unmake

Build the `Incremental` design below when a real game shows either of these:

- Copying the position, without the evaluation update, is more than 20% of the
  instructions per node under callgrind, measured as above against an in-place variant.
- The position cannot be cloned cheaply at all, for example because it holds large
  buffers that a move changes only in a few places.

## Design on file: `Incremental`

If it is built, it should look like this.

### API

```rust
/// A game whose positions can be changed in place and changed back, so the alpha-beta
/// family can search with one position instead of one per move.
pub trait Incremental: Game {
    /// What `make` needs to undo a move: captured pieces, castling rights, the previous
    /// hash, and so on.
    type Undo;

    /// Plays `mv` on `position` in place, returning what `unmake` needs.
    fn make(&self, position: &mut Self::Position, mv: &Self::Move) -> Self::Undo;

    /// Takes `mv` back, restoring `position` exactly as it was before `make`.
    fn unmake(&self, position: &mut Self::Position, mv: &Self::Move, undo: Self::Undo);
}
```

- `Game::play` stays required. MCTS, expectimax, `minimax` and `negamax` keep using it, since
  they need positions they can keep or compare. A game can implement `play` as clone plus
  `make`.
- The entry points are `Searcher::search_in_place` and `Searcher::deepen_in_place`, with
  `G: Incremental` and `G::Position: Clone`. They take `&G::Position` like the others and
  clone it once into the searcher. The caller's position is never mutated, so a panic or a
  budget abort cannot leave it half-played. Lazy SMP needs one position per thread anyway.

### Inside the searcher

- All alpha-beta-family searches play moves in one place, `Searcher::expand`, which builds
  the child with `game.play` and visits it. PVS and MTD(f) share that code.
- An internal "current line" type would replace the position parameter of `visit`,
  `expand` and `quiesce`:
  - **copy:** a stack of owned positions; descending pushes `play(top, mv)`, returning pops.
  - **in place:** one position and a stack of `Undo` values; descending calls `make`,
    returning calls `unmake`.
- Moves are listed into the ply's move buffer before the loop, so the position is not
  borrowed while it changes.
- Nothing else depends on positions:
  - The transposition key comes from `Hashed::key` on the current position. A game updates
    an incremental Zobrist hash inside `make` and `unmake`.
  - The principal variation, the hint, the root move report and the table's best move are
    all moves.
  - Quiescence descends through the same line.
  - A PVS re-search or an aspiration re-search starts again from the same position, which
    `unmake` restored.

### Tests it would need

- In-place and copy searches give equal results, statistics included, for every
  algorithm, with and without a table and with quiescence.
- After a search, after a budget abort, and after PVS and aspiration re-searches, the
  searcher's position equals the root again. A `make` without its `unmake` would show
  there.
- Callgrind on the `deepen/*` benches shows that the copy path did not get slower, since
  the "current line" changes the hot path of every search.

## Further options

### Probing before play

`visit` probes the table only after the child exists: `play`, then `moves` and
`order_moves`, then the probe. An optional hook such as `Hashed::key_after(parent_key, mv)`
would let `expand` probe a child's entry before playing the move, and skip `play`, and the
accumulator delta with it, on a hit that answers the search. Before building it, check:

- that probing earlier keeps the hit counts and the results equal;
- the interplay with the hint, which needs the child's frame;
- how much a real game gains.

### Lazy accumulators

Engines can also update lazily: `play` records only which features changed, and `evaluate`
walks back to the last ancestor with a computed accumulator. With copy-make, a child has
no way back to its parent, so lazy updating needs a game-owned stack. The probe's stack
variant shows that a game can already keep one, indexed by ply. That is sound only because
the searcher plays each child just before searching it, depth first, and that order is
not part of the API today. Supporting lazy accumulators would mean either guaranteeing that
order for the alpha-beta family, or giving searches a way to tell the game the ply. Both
are open. Neither works across Lazy SMP threads without per-thread state.

## Open questions

- Is a real network user available to validate the 20% trigger on a real engine?
- Should the copy-make pattern be shown as a compiling doctest with a toy accumulator, or
  in prose only?
- Is the depth-first `play` order worth guaranteeing for the alpha-beta family, so a game
  may keep per-ply state?
