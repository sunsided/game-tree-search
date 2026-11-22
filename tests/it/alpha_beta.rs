//! Cross-checks of alpha-beta against plain minimax, with and without satisficing.

use game_tree_search::{Bound, Game, SearchResult, Searcher, Side, alpha_beta, minimax};

use crate::agreement::replayed;
use crate::algorithms::{ALGORITHMS, search};
use crate::games::{Nim, RandomTree};

/// Seeds of the random trees the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// Checks what every alpha-beta result promises, satisficing or not: the root score admits
/// the minimax value, the principal variation replays to the score, and every root move
/// admits the minimax value of the position it leads to.
fn assert_sound(game: &RandomTree, seed: u64, depth: u32, result: &SearchResult<u64, i32>) {
    let reference = minimax(game, &seed, depth);
    assert!(
        result.bound.admits(result.score, reference.score),
        "seed {seed}, depth {depth}"
    );
    assert_eq!(
        replayed(game, &seed, result),
        result.score,
        "seed {seed}, depth {depth}"
    );
    assert_eq!(
        result.root_moves.len() + result.unexamined,
        reference.root_moves.len(),
        "seed {seed}, depth {depth}"
    );

    let mut position = seed;
    for (ply, step) in (1..).zip(&result.pv) {
        position = game.play(&position, &step.mv);
        let value = minimax(game, &position, depth - ply).score;
        assert!(
            step.bound.admits(step.score, value),
            "seed {seed}, ply {ply}"
        );
    }

    for root_move in &result.root_moves {
        let child = game.play(&seed, &root_move.mv);
        let value = minimax(game, &child, depth - 1).score;
        assert!(
            root_move.bound.admits(root_move.score, value),
            "seed {seed}, move {}",
            root_move.mv
        );
    }
}

/// Without satisficing, alpha-beta returns minimax's exact score, in either move order.
#[test]
fn alpha_beta_matches_minimax_on_random_trees() {
    for algorithm in ALGORITHMS {
        for reversed in [false, true] {
            let game = RandomTree {
                reversed,
                ..RandomTree::new(4)
            };
            for seed in SEEDS {
                for depth in 0..=6 {
                    let result = search(algorithm, &game, &seed, depth);
                    assert_sound(&game, seed, depth, &result);
                    assert_eq!(result.bound, Bound::Exact, "seed {seed}, depth {depth}");
                    assert_eq!(result.score, minimax(&game, &seed, depth).score);
                    assert_eq!(result.unexamined, 0);
                }
            }
        }
    }
}

/// The best root move is scored exactly, like the root.
#[test]
fn the_best_root_move_is_exact() {
    for algorithm in ALGORITHMS {
        let game = RandomTree::new(4);
        for seed in SEEDS {
            let result = search(algorithm, &game, &seed, 5);
            let Some(best) = result.best() else { continue };
            let root_move = result
                .root_moves
                .iter()
                .find(|root_move| root_move.mv == *best);
            let root_move = root_move.expect("the best move was searched");
            assert_eq!(
                (root_move.score, root_move.bound),
                (result.score, Bound::Exact)
            );
        }
    }
}

/// A reused searcher returns exactly what a fresh one returns, search after search.
#[test]
fn a_reused_searcher_matches_fresh_searches() {
    for algorithm in ALGORITHMS {
        let game = RandomTree {
            min_satisfied_below: Some(0),
            ..RandomTree::new(4)
        };
        let mut searcher = Searcher::default().with_algorithm(algorithm);
        for seed in SEEDS {
            for depth in [5, 0, 3, 6] {
                let reused = searcher.search(&game, &seed, depth);
                assert_eq!(
                    reused,
                    search(algorithm, &game, &seed, depth),
                    "seed {seed}, depth {depth}"
                );
            }
        }
    }
}

/// Alpha-beta visits fewer positions than minimax over the random trees.
#[test]
fn alpha_beta_prunes() {
    let game = RandomTree::new(4);
    let nodes = |search: fn(&RandomTree, &u64, u32) -> SearchResult<u64, i32>| {
        SEEDS
            .map(|seed| search(&game, &seed, 6).stats.nodes)
            .sum::<u64>()
    };
    assert!(nodes(alpha_beta) < nodes(minimax));
}

/// With satisficing, results stay sound while positions are cut and root scores loosen to
/// bounds.
#[test]
fn satisficing_results_admit_the_minimax_value() {
    for algorithm in ALGORITHMS {
        let game = RandomTree {
            min_satisfied_below: Some(0),
            ..RandomTree::new(4)
        };
        let mut loosened = 0;
        let mut cuts = 0;
        for seed in SEEDS {
            for depth in 1..=6 {
                let result = search(algorithm, &game, &seed, depth);
                assert_sound(&game, seed, depth, &result);
                cuts += result.stats.cuts;
                loosened += usize::from(result.bound != Bound::Exact);
            }
        }

        assert!(cuts > 0);
        assert!(loosened > 0);
    }
}

/// When both sides satisfice on overlapping ranges, bounds that pull in opposite
/// directions meet, and some results become approximate; they stay sound.
#[test]
fn two_sided_satisficing_yields_approximate_bounds() {
    for algorithm in ALGORITHMS {
        let game = RandomTree {
            min_satisfied_below: Some(0),
            max_satisfied_above: Some(-10),
            ..RandomTree::new(4)
        };
        let mut approximate = 0;
        for seed in SEEDS {
            for depth in 1..=6 {
                let result = search(algorithm, &game, &seed, depth);
                assert_sound(&game, seed, depth, &result);
                approximate += usize::from(result.bound == Bound::Approximate);
            }
        }

        assert!(approximate > 0);
    }
}

/// A satisfied root skips its remaining moves and reports them as unexamined.
#[test]
fn a_satisfied_root_reports_unexamined_moves() {
    for algorithm in ALGORITHMS {
        let game = RandomTree {
            min_satisfied_below: Some(101),
            ..RandomTree::new(4)
        };
        let root = SEEDS
            .into_iter()
            .find(|seed| {
                let mut moves = Vec::new();
                game.moves(seed, &mut moves);
                moves.len() > 1 && game.side_to_move(seed) == Side::Min
            })
            .expect("some root has several moves and Min to move");

        let result = search(algorithm, &game, &root, 3);
        assert_eq!(result.root_moves.len(), 1);
        assert!(result.unexamined > 0);
        assert!(matches!(result.bound, Bound::Upper | Bound::Approximate));
    }
}

/// On Nim, alpha-beta agrees with minimax at every pile size.
#[test]
fn alpha_beta_solves_nim() {
    for algorithm in ALGORITHMS {
        for stones in 0..=12 {
            for side in [Side::Max, Side::Min] {
                let root = (stones, side);
                let result = search(algorithm, &Nim, &root, 12);
                assert_eq!(
                    (result.score, result.bound),
                    (minimax(&Nim, &root, 12).score, Bound::Exact)
                );
            }
        }
    }
}
