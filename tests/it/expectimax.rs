//! Cross-checks of expectimax: minimax without chance, means at chance positions, and
//! values along the principal variation.

use game_tree_search::{Bound, Game, Stochastic, TotalF64, expectimax, minimax};

use crate::games::{Dicey, RandomTree};

/// Seeds of the random trees the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// Random trees with one chance position in `chance`, weights scaled by `scale`.
fn dicey(chance: u64, scale: f64, noisy: bool) -> Dicey {
    Dicey {
        tree: RandomTree {
            noisy,
            ..RandomTree::new(4)
        },
        chance,
        scale,
    }
}

/// Whether two scores agree up to float rounding.
fn close(a: TotalF64, b: TotalF64) -> bool {
    (a.get() - b.get()).abs() < 1e-9
}

/// Without chance positions, expectimax is minimax, quiescence included.
#[test]
fn without_chance_expectimax_is_minimax() {
    for noisy in [false, true] {
        let game = dicey(0, 1.0, noisy);
        for seed in SEEDS {
            for depth in 0..=4 {
                assert_eq!(
                    expectimax(&game, &seed, depth),
                    minimax(&game, &seed, depth)
                );
            }
        }
    }
}

/// A chance root is worth the weighted mean of its outcomes, a choice root its best move;
/// every result is exact.
#[test]
fn roots_are_means_or_best_moves() {
    let game = dicey(3, 1.0, false);
    let mut chance_roots = 0;
    for seed in SEEDS {
        let result = expectimax(&game, &seed, 4);
        assert_eq!(result.bound, Bound::Exact);
        let mut outcomes = Vec::new();
        game.outcomes(&seed, &mut outcomes);
        if outcomes.is_empty() {
            if let Some(best) = result.best() {
                let found = result
                    .root_moves
                    .iter()
                    .find(|root_move| root_move.mv == *best);
                assert!(
                    close(found.expect("searched").score, result.score),
                    "seed {seed}"
                );
            }

            continue;
        }

        chance_roots += 1;
        let total: f64 = outcomes.iter().map(|&(_, weight)| weight).sum();
        let mean: f64 = result
            .root_moves
            .iter()
            .zip(&outcomes)
            .map(|(root_move, &(_, weight))| root_move.score.get() * weight)
            .sum::<f64>()
            / total;
        assert!(
            close(result.score, TotalF64::new(mean).expect("finite")),
            "seed {seed}"
        );
    }

    assert!(chance_roots > 0);
}

/// Every step of the principal variation carries the expectimax value of the position it
/// leads to, and chance lines follow the most likely outcome.
#[test]
fn line_steps_carry_their_positions_values() {
    let game = dicey(3, 1.0, true);
    for seed in SEEDS {
        let result = expectimax(&game, &seed, 4);
        let mut position = seed;
        for (ply, step) in (1..).zip(&result.pv) {
            let mut outcomes = Vec::new();
            game.outcomes(&position, &mut outcomes);
            if let Some(likeliest) = outcomes.iter().map(|&(_, weight)| weight).reduce(f64::max) {
                let weight = outcomes
                    .iter()
                    .find(|&&(mv, _)| mv == step.mv)
                    .map(|&(_, w)| w);
                assert_eq!(weight, Some(likeliest), "seed {seed}, ply {ply}");
            }

            position = game.play(&position, &step.mv);
            let remaining = 4_u32.saturating_sub(ply);
            let value = expectimax(&game, &position, remaining).score;
            assert!(close(step.score, value), "seed {seed}, ply {ply}");
        }
    }
}

/// Only the ratios of outcome weights matter.
#[test]
fn outcome_weights_need_not_sum_to_one() {
    let (plain, scaled) = (dicey(3, 1.0, false), dicey(3, 7.5, false));
    for seed in SEEDS {
        let (a, b) = (expectimax(&plain, &seed, 4), expectimax(&scaled, &seed, 4));
        assert!(close(a.score, b.score), "seed {seed}");
    }
}
