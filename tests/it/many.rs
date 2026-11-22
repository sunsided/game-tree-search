//! Many searches at once: searchers are `Send`, games are shared by reference, and one
//! searcher per thread returns what it returns on one thread.

use std::thread;

use game_tree_search::{Mcts, Searcher, SlotTable, Stats, TreeStats};

use crate::games::{Nim, RandomTree};

/// Compiles only if `T` can move to another thread.
const fn assert_send<T: Send>() {}

/// Compiles only if `&T` can be shared between threads.
const fn assert_sync<T: Sync>() {}

/// Searchers, with or without a table, and Monte Carlo searches can move between threads,
/// and the test games can be shared between them.
#[test]
fn searchers_are_send_and_games_are_sync() {
    assert_send::<Searcher<RandomTree>>();
    assert_send::<Searcher<RandomTree, game_tree_search::Transpositions<SlotTable<u64, u64, i32>>>>(
    );
    assert_send::<Mcts<Nim>>();
    assert_sync::<RandomTree>();
    assert_sync::<Nim>();
}

/// Four threads share one game, each with its own searcher, and each returns exactly what
/// the same searches return on one thread.
#[test]
fn one_searcher_per_thread_shares_the_game() {
    let game = RandomTree {
        width: Some(3),
        ..RandomTree::new(4)
    };
    let roots: Vec<Vec<u64>> = (0..4)
        .map(|agent| (agent * 50..agent * 50 + 50).collect())
        .collect();
    let run = |roots: &[u64]| {
        let mut searcher = Searcher::with_table(SlotTable::new(1 << 10));
        let mut mcts = Mcts::new();
        roots
            .iter()
            .map(|root| {
                let deep = searcher.deepen(&game, root, 5, |_: &Stats| false);
                let simulated =
                    mcts.search(&game, root, |stats: &TreeStats| stats.simulations >= 50);
                (deep, simulated)
            })
            .collect::<Vec<_>>()
    };

    let parallel: Vec<_> = thread::scope(|scope| {
        let handles: Vec<_> = roots
            .iter()
            .map(|roots| scope.spawn(|| run(roots)))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("no panic"))
            .collect()
    });
    let sequential: Vec<_> = roots.iter().map(|roots| run(roots)).collect();
    assert_eq!(parallel, sequential);
}
