//! Lazy SMP over a shared table (feature `parallel`): threads share entries, and the score
//! matches a single-threaded search.
#![cfg(feature = "parallel")]

use game_tree_search::{
    Algorithm, Bound, Game, Hashed, Searcher, SharedTable, Side, Stats, deepen_shared, minimax,
};

use crate::algorithms::ALGORITHMS;
use crate::games::RandomTree;

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
