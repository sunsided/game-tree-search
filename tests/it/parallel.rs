//! Root-parallel Monte Carlo tree search (feature `parallel`): merged statistics are the
//! sum of independent searches, reproducibly.
#![cfg(feature = "parallel")]

use game_tree_search::{Mcts, Random, Side, SplitMix64, TreeStats};

use crate::games::{Nim, RandomTree};

/// A budget of `n` simulations.
fn simulations(n: u64) -> impl FnMut(&TreeStats) -> bool + Clone + Send {
    move |stats: &TreeStats| stats.simulations >= n
}

/// The parallel result is the sum of the searches its threads run, each seeded from the
/// searcher's generator in turn.
#[test]
fn merged_statistics_add_up_the_threads() {
    let game = RandomTree::new(4);
    for root in 0..30 {
        let merged = Mcts::new().search_parallel(&game, &root, 3, simulations(100));
        let mut seeds = SplitMix64::new(0);
        let singles: Vec<_> = (0..3)
            .map(|_| {
                let mut single = Mcts::new().with_rng(SplitMix64::new(seeds.next_u64()));
                single.search(&game, &root, simulations(100))
            })
            .collect();

        assert_eq!(merged.stats.simulations, 300, "root {root}");
        for (index, root_move) in merged.root.iter().enumerate() {
            let visits: u32 = singles.iter().map(|single| single.root[index].visits).sum();
            assert_eq!(root_move.visits, visits, "root {root}");
        }

        let most = merged.root.iter().map(|root_move| root_move.visits).max();
        let best = merged
            .root
            .iter()
            .find(|root_move| Some(&root_move.mv) == merged.best.as_ref());
        assert_eq!(best.map(|root_move| root_move.visits), most, "root {root}");
        assert_eq!(merged.pv.first(), merged.best.as_ref(), "root {root}");
    }
}

/// One thread, or zero, runs one search seeded from the searcher's first draw.
#[test]
fn one_thread_is_one_search() {
    let game = RandomTree::new(4);
    let seed = SplitMix64::new(0).next_u64();
    for root in 0..20 {
        let single =
            Mcts::new()
                .with_rng(SplitMix64::new(seed))
                .search(&game, &root, simulations(200));
        for threads in [0, 1] {
            let parallel = Mcts::new().search_parallel(&game, &root, threads, simulations(200));
            assert_eq!(parallel, single, "root {root}, {threads} threads");
        }
    }
}

/// The same seed reproduces the same merged result, and the threads together solve Nim.
#[test]
fn parallel_searches_are_reproducible_and_solve_nim() {
    let search = || Mcts::new().search_parallel(&Nim, &(10, Side::Max), 4, simulations(1000));
    let result = search();
    assert_eq!(result, search());
    assert_eq!(result.best, Some(1));

    let finished = Mcts::new().search_parallel(&Nim, &(0, Side::Max), 2, simulations(5));
    assert_eq!((finished.best, finished.root.len()), (None, 0));
}

/// A game whose every move panics: a test-only game.
struct Exploding;

impl game_tree_search::Game for Exploding {
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

impl game_tree_search::Rewarded for Exploding {
    fn reward(&self, _: i32) -> f64 {
        0.5
    }
}

/// A panic in a thread reaches the caller with its own message.
#[test]
#[should_panic(expected = "the game exploded")]
fn a_panic_in_a_thread_reaches_the_caller() {
    let _ = Mcts::new().search_parallel(&Exploding, &(), 2, simulations(10));
}

/// The merged root value is the mean over all trees' simulations.
#[test]
fn merged_values_average_the_trees() {
    let merged = Mcts::new().search_parallel(&Nim, &(7, Side::Max), 3, simulations(500));
    let mut seeds = SplitMix64::new(0);
    let mean = (0..3)
        .map(|_| {
            Mcts::new()
                .with_rng(SplitMix64::new(seeds.next_u64()))
                .search(&Nim, &(7, Side::Max), simulations(500))
                .value
        })
        .sum::<f64>()
        / 3.0;
    assert!(
        (merged.value - mean).abs() < 1e-9,
        "{} vs {mean}",
        merged.value
    );
}

/// Threads sharing one tree run exactly the budgeted simulations between them, every one
/// through the root, and solve Nim.
#[test]
fn shared_trees_run_exactly_the_budget() {
    let game = RandomTree::new(4);
    for threads in [1, 2, 4, 8] {
        for root in 0..10 {
            let result = Mcts::new().search_shared(&game, &root, threads, simulations(300));
            assert_eq!(
                result.stats.simulations, 300,
                "{threads} threads, root {root}"
            );
            if !result.root.is_empty() {
                let visits: u32 = result.root.iter().map(|root_move| root_move.visits).sum();
                assert_eq!(visits, 300, "{threads} threads, root {root}");
            }
        }
    }

    let nim = Mcts::new().search_shared(&Nim, &(10, Side::Max), 4, simulations(2000));
    assert_eq!(nim.best, Some(1));
}

/// One thread, or zero, on a shared tree is one search with the generator seeded from the
/// searcher's first draw: nothing else ever waits, so virtual loss changes nothing.
#[test]
fn one_thread_on_a_shared_tree_is_one_search() {
    let game = RandomTree::new(4);
    let seed = SplitMix64::new(0).next_u64();
    for root in 0..20 {
        let single =
            Mcts::new()
                .with_rng(SplitMix64::new(seed))
                .search(&game, &root, simulations(200));
        for threads in [0, 1] {
            let shared = Mcts::new().search_shared(&game, &root, threads, simulations(200));
            assert_eq!(shared, single, "root {root}, {threads} threads");
        }
    }
}

/// A shared search continues a tree kept by `advance`, and a finished root has no move.
#[test]
fn shared_searches_continue_kept_trees() {
    let mut mcts = Mcts::new();
    let first = mcts.search_shared(&Nim, &(9, Side::Max), 3, simulations(400));
    let taken = first.best.expect("nine stones have moves");
    assert!(mcts.advance(&[taken]));

    let next = mcts.search_shared(&Nim, &(9 - taken, Side::Min), 3, simulations(100));
    let visits: u32 = next.root.iter().map(|root_move| root_move.visits).sum();
    assert!(visits > 100, "{visits}");
    assert_eq!(next.stats.simulations, 100);

    let finished = Mcts::new().search_shared(&Nim, &(0, Side::Max), 2, simulations(5));
    assert_eq!((finished.best, finished.root.len()), (None, 0));
}

/// A panic in a thread sharing the tree reaches the caller with its own message.
#[test]
#[should_panic(expected = "the game exploded")]
fn a_panic_on_a_shared_tree_reaches_the_caller() {
    let _ = Mcts::new().search_shared(&Exploding, &(), 2, simulations(10));
}
