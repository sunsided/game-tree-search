//! Lazy SMP over a shared table (feature `parallel`): threads share entries, and the score
//! matches a single-threaded search.
#![cfg(feature = "parallel")]

use game_tree_search::{
    Algorithm, Bound, Game, Hashed, PackedTable, Searcher, SharedTable, Side, SlotTable, Stats,
    deepen_shared, minimax,
};

use crate::algorithms::ALGORITHMS;
use crate::games::{Narrow, RandomTree, Smooth};

/// Seeds of the random games the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..100;

/// Random games where each ply has only three positions, so threads meet in the table.
fn narrow() -> RandomTree {
    RandomTree {
        width: Some(3),
        ..RandomTree::new(4)
    }
}

/// A budget that never runs out.
fn unlimited(_: &Stats) -> bool {
    false
}

/// Four threads sharing a table return the exact minimax score at the depth reached, for
/// every algorithm, and leave entries for the next search.
#[test]
fn lazy_smp_matches_minimax() {
    let game = narrow();
    for algorithm in ALGORITHMS {
        let table = SharedTable::new(1 << 12, 8);
        for seed in SEEDS {
            let result = deepen_shared(&game, &seed, 6, (4, algorithm), &table, unlimited);
            let reference = minimax(&game, &seed, result.depth).score;
            assert_eq!(
                (result.score, result.bound),
                (reference, Bound::Exact),
                "{algorithm:?}, seed {seed}"
            );
            assert!(
                result.depth == 6 || result.proven,
                "{algorithm:?}, seed {seed}"
            );
        }

        assert!(table.capacity() >= 1 << 12);
    }
}

/// Four threads sharing a lock-free [`PackedTable`] return the exact minimax score at the
/// depth reached, for every algorithm.
#[test]
fn lazy_smp_on_a_packed_table_matches_minimax() {
    let game = Narrow(narrow());
    for algorithm in ALGORITHMS {
        let table = PackedTable::new(1 << 12);
        for seed in SEEDS {
            let result = deepen_shared(&game, &seed, 6, (4, algorithm), &table, unlimited);
            let reference = minimax(&game, &seed, result.depth).score;
            assert_eq!(
                (result.score, result.bound),
                (reference, Bound::Exact),
                "{algorithm:?}, seed {seed}"
            );
        }
    }
}

/// On one thread, a [`PackedTable`] keeps and returns the same entries as a [`SlotTable`]
/// of the same size, so the searches agree to the last counter: the packing loses nothing.
#[test]
fn a_packed_table_stores_what_a_slot_table_stores() {
    let game = Narrow(narrow());
    for seed in SEEDS {
        let packed = PackedTable::new(1 << 10);
        let packed = Searcher::with_table(&packed).deepen(&game, &seed, 8, unlimited);
        let slot = Searcher::with_table(SlotTable::new(1 << 10)).deepen(&game, &seed, 8, unlimited);
        assert_eq!(packed, slot, "seed {seed}");
        assert!(packed.stats.table_hits > 0 || packed.proven, "seed {seed}");
    }
}

/// One thread, or zero, is plain deepening with the shared table.
#[test]
fn one_thread_is_plain_deepening() {
    let game = narrow();
    for threads in [0, 1] {
        for seed in SEEDS {
            let (shared, plain) = (SharedTable::new(1 << 10, 4), SharedTable::new(1 << 10, 4));
            let lazy = deepen_shared(
                &game,
                &seed,
                5,
                (threads, Algorithm::Pvs),
                &shared,
                unlimited,
            );
            let single = Searcher::with_table(&plain)
                .with_algorithm(Algorithm::Pvs)
                .deepen(&game, &seed, 5, unlimited);
            assert_eq!(lazy, single, "{threads} threads, seed {seed}");
        }
    }
}

/// The main thread's budget stops all threads; the result is its last finished iteration.
#[test]
fn the_main_budget_stops_every_thread() {
    let game = narrow();
    let table = SharedTable::new(1 << 12, 8);
    for seed in SEEDS {
        table.clear();
        let budget = |stats: &Stats| stats.nodes >= 100;
        let result = deepen_shared(&game, &seed, 20, (3, Algorithm::AlphaBeta), &table, budget);
        assert_eq!(
            result.score,
            minimax(&game, &seed, result.depth).score,
            "seed {seed}"
        );
        assert!(result.depth < 20, "seed {seed}");
    }
}

/// A helper stopped in its first iteration, which the budget asks there, has no answer: a
/// shallow search with a budget that runs out at once still returns a real line, never the
/// helper's bare root evaluation at the full depth.
#[test]
fn a_helper_stopped_at_once_never_answers() {
    let game = narrow();
    let table = SharedTable::new(1 << 12, 8);
    for seed in SEEDS {
        table.clear();
        let budget = |stats: &Stats| stats.nodes >= 1;
        let result = deepen_shared(&game, &seed, 2, (2, Algorithm::AlphaBeta), &table, budget);
        let mut moves = Vec::new();
        game.moves(&seed, &mut moves);
        assert!(result.stats.nodes > 0, "seed {seed}");
        assert_eq!(result.best().is_some(), !moves.is_empty(), "seed {seed}");
        assert_eq!(
            result.score,
            minimax(&game, &seed, result.depth).score,
            "seed {seed}"
        );
    }
}

/// A game whose every move panics: a test-only game.
struct Exploding;

impl Game for Exploding {
    /// A single position.
    type Position = ();
    /// A single move.
    type Move = ();
    /// Always `0`.
    type Score = i32;

    fn side_to_move(&self, (): &()) -> Side {
        Side::Max
    }

    fn moves(&self, (): &(), moves: &mut Vec<()>) {
        moves.push(());
    }

    fn play(&self, (): &(), (): &()) {
        panic!("the game exploded");
    }

    fn evaluate(&self, (): &()) -> i32 {
        0
    }
}

impl Hashed for Exploding {
    /// The single position's key.
    type Key = u64;

    fn key(&self, (): &()) -> u64 {
        0
    }
}

/// A panic in the main thread reaches the caller after the helpers stopped.
#[test]
#[should_panic(expected = "the game exploded")]
fn a_panic_reaches_the_caller() {
    let table = SharedTable::new(16, 2);
    let _ = deepen_shared(
        &Exploding,
        &(),
        3,
        (2, Algorithm::AlphaBeta),
        &table,
        unlimited,
    );
}

/// A panic in the main thread reaches the caller when it searches alone.
#[test]
#[should_panic(expected = "the game exploded")]
fn a_panic_in_the_main_thread_reaches_the_caller() {
    let table = SharedTable::new(16, 2);
    let _ = deepen_shared(
        &Exploding,
        &(),
        3,
        (1, Algorithm::AlphaBeta),
        &table,
        unlimited,
    );
}

/// A [`Narrow`] tree searches exactly like the tree it wraps, quiescence included, so the
/// packed-table tests and benches play the same games as the rest.
#[test]
fn a_narrow_tree_searches_like_its_tree() {
    let tree = |noisy| RandomTree {
        noisy,
        ..RandomTree::new(4)
    };
    let mut quiesced = 0;
    for seed in SEEDS {
        let wide = Searcher::new().search(&tree(true), &seed, 4);
        let narrow = Searcher::new().search(&Narrow(tree(true)), &seed, 4);
        assert_eq!(
            (narrow.score, narrow.stats),
            (wide.score, wide.stats),
            "seed {seed}"
        );
        let best = narrow.best().map(|&mv| u64::from(mv));
        assert_eq!(best, wide.best().copied(), "seed {seed}");
        let quiet = Searcher::new().search(&tree(false), &seed, 4);
        quiesced += usize::from(quiet.stats != wide.stats);
    }

    // Quiescence searched past the horizon somewhere, so the noisy moves were forwarded.
    assert!(quiesced > 0);
}

/// On trees whose move ordering matters, Lazy SMP helpers that keep the game's first move
/// still return minimax's exact score, for every algorithm.
#[test]
fn lazy_smp_on_ordered_trees_matches_minimax() {
    let game = Smooth {
        tree: RandomTree::new(4),
        ordered: true,
    };
    for algorithm in ALGORITHMS {
        for seed in SEEDS {
            let root = (seed, 0, 0);
            let table = PackedTable::new(1 << 12);
            let result = deepen_shared(&game, &root, 5, (4, algorithm), &table, unlimited);
            let reference = minimax(&game, &root, result.depth).score;
            assert_eq!(
                (result.score, result.bound),
                (reference, Bound::Exact),
                "{algorithm:?}, seed {seed}"
            );
        }
    }
}
