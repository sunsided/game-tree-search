//! Cross-checks of iterative deepening against single alpha-beta and minimax searches.

use game_tree_search::{Bound, Searcher, Side, Stats, alpha_beta, deepen, minimax};

use crate::agreement::replayed;
use crate::games::{Nim, RandomTree};

/// Seeds of the random trees the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// A budget that never runs out.
fn unlimited(_: &Stats) -> bool {
    false
}

/// Without a budget, deepening returns the score of one alpha-beta search to the depth it
/// reached, and reaches `max_depth` unless the game was searched to its end first; a proven
/// score holds for deeper searches too.
#[test]
fn unlimited_deepening_matches_alpha_beta() {
    let game = RandomTree::new(4);
    for seed in SEEDS {
        for max_depth in 0..=6 {
            let result = deepen(&game, &seed, max_depth, unlimited);
            let reference = alpha_beta(&game, &seed, result.depth);
            assert_eq!(
                (result.score, result.bound),
                (reference.score, Bound::Exact),
                "seed {seed}"
            );
            assert_eq!(replayed(&game, &seed, &result), result.score, "seed {seed}");
            assert!(result.depth == max_depth || result.proven, "seed {seed}");
            if result.proven {
                let deeper = alpha_beta(&game, &seed, result.depth + 2);
                assert_eq!(deeper.score, result.score, "seed {seed}");
            }
        }
    }
}

/// With satisficing, deepening stays sound: the score admits the minimax value at the depth
/// reached.
#[test]
fn satisficing_deepening_admits_the_minimax_value() {
    let game = RandomTree {
        min_satisfied_below: Some(0),
        max_satisfied_above: Some(-10),
        ..RandomTree::new(4)
    };
    for seed in SEEDS {
        let result = deepen(&game, &seed, 6, unlimited);
        let value = minimax(&game, &seed, result.depth).score;
        assert!(result.bound.admits(result.score, value), "seed {seed}");
        assert_eq!(replayed(&game, &seed, &result), result.score, "seed {seed}");
    }
}

/// A node budget stops deepening early; the result is the last iteration that finished.
#[test]
fn a_node_budget_keeps_the_last_finished_iteration() {
    let game = RandomTree::new(4);
    let mut stopped = 0;
    for seed in SEEDS {
        let result = deepen(&game, &seed, 12, |stats: &Stats| stats.nodes >= 200);
        let reference = alpha_beta(&game, &seed, result.depth);
        assert_eq!(result.score, reference.score, "seed {seed}");
        assert!(
            result.stats.nodes <= 200 || result.depth == 1,
            "seed {seed}"
        );
        stopped += usize::from(result.depth < 12 && !result.proven);
    }

    assert!(stopped > 0);
}

/// The first iteration ignores the budget, so even an exhausted budget yields a move.
#[test]
fn the_first_iteration_always_finishes() {
    let result = deepen(&Nim, &(10, Side::Max), 10, |_: &Stats| true);
    assert_eq!(result.depth, 1);
    assert!(result.best().is_some());
}

/// Deepening stops as soon as the whole game has been searched.
#[test]
fn deepening_stops_once_proven() {
    let result = deepen(&Nim, &(5, Side::Max), 40, unlimited);
    assert!(result.proven);
    assert_eq!((result.depth, result.score), (6, 1));
}

/// A maximum depth of zero evaluates the root.
#[test]
fn depth_zero_evaluates_the_root() {
    let result = deepen(&Nim, &(5, Side::Max), 0, unlimited);
    assert_eq!((result.depth, result.score, result.best()), (0, 0, None));
    assert_eq!(result.stats.horizon, 1);
}

/// A reused searcher deepens exactly like a fresh one.
#[test]
fn a_reused_searcher_deepens_like_a_fresh_one() {
    let game = RandomTree::new(4);
    let mut searcher = Searcher::new();
    for seed in SEEDS {
        let reused = searcher.deepen(&game, &seed, 5, unlimited);
        assert_eq!(reused, deepen(&game, &seed, 5, unlimited), "seed {seed}");
    }
}

/// The reference searches report whether the game was searched to its end.
#[test]
fn reference_searches_report_proven_results() {
    assert!(minimax(&Nim, &(4, Side::Max), 5).proven);
    assert!(!minimax(&Nim, &(4, Side::Max), 4).proven);
}
