//! Cross-checks of quiescence search: every algorithm searches noisy moves past the horizon
//! exactly as the reference searches do.

use game_tree_search::{Bound, Searcher, SlotTable, Stats, minimax, negamax};

use crate::agreement::replayed;
use crate::algorithms::{ALGORITHMS, deepen_with_algorithm, search};
use crate::games::RandomTree;

/// Seeds of the random trees the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// Random trees with noisy moves.
fn noisy() -> RandomTree {
    RandomTree {
        noisy: true,
        ..RandomTree::new(4)
    }
}

/// A budget that never runs out.
fn unlimited(_: &Stats) -> bool {
    false
}

/// Quiescence changes scores and runs lines past the depth searched, and negamax agrees with
/// minimax on it.
#[test]
fn quiescence_extends_the_reference_searches() {
    let (quiet, game) = (RandomTree::new(4), noisy());
    let (mut changed, mut extended) = (0, 0);
    for seed in SEEDS {
        for depth in 0..=4 {
            let result = minimax(&game, &seed, depth);
            assert_eq!(
                negamax(&game, &seed, depth),
                result,
                "seed {seed}, depth {depth}"
            );
            assert_eq!(replayed(&game, &seed, &result), result.score, "seed {seed}");
            changed += usize::from(result.score != minimax(&quiet, &seed, depth).score);
            extended += usize::from(result.pv.len() > usize::try_from(depth).expect("small"));
        }
    }

    assert!(
        changed > 0 && extended > 0,
        "{changed} changed, {extended} extended"
    );
}

/// Every algorithm returns the exact quiescence score of the reference, with a line that
/// replays to it.
#[test]
fn every_algorithm_matches_the_reference_quiescence() {
    let game = noisy();
    for algorithm in ALGORITHMS {
        for seed in SEEDS {
            for depth in 0..=5 {
                let result = search(algorithm, &game, &seed, depth);
                let reference = minimax(&game, &seed, depth).score;
                assert_eq!(
                    (result.score, result.bound),
                    (reference, Bound::Exact),
                    "{algorithm:?}"
                );
                assert_eq!(
                    replayed(&game, &seed, &result),
                    result.score,
                    "{algorithm:?}"
                );
            }
        }
    }
}

/// Quiescence under two-sided satisficing stays sound.
#[test]
fn satisficing_quiescence_admits_the_minimax_value() {
    let game = RandomTree {
        min_satisfied_below: Some(0),
        max_satisfied_above: Some(-10),
        ..noisy()
    };
    for algorithm in ALGORITHMS {
        for seed in SEEDS {
            for depth in 0..=4 {
                let result = search(algorithm, &game, &seed, depth);
                let value = minimax(&game, &seed, depth).score;
                assert!(
                    result.bound.admits(result.score, value),
                    "{algorithm:?}, seed {seed}"
                );
            }
        }
    }
}

/// Quiescence with a transposition table, and under deepening, stays exact.
#[test]
fn quiescence_with_a_table_and_deepening_stays_exact() {
    let game = RandomTree {
        width: Some(3),
        ..noisy()
    };
    for algorithm in ALGORITHMS {
        let mut searcher = Searcher::with_table(SlotTable::new(1 << 12)).with_algorithm(algorithm);
        for seed in SEEDS {
            let tabled = searcher.deepen(&game, &seed, 5, unlimited);
            let plain = deepen_with_algorithm(algorithm, &game, &seed, 5, unlimited);
            let reference = minimax(&game, &seed, tabled.depth).score;
            assert_eq!(
                (tabled.score, tabled.bound),
                (reference, Bound::Exact),
                "seed {seed}"
            );
            assert_eq!(
                plain.score,
                minimax(&game, &seed, plain.depth).score,
                "seed {seed}"
            );
        }
    }
}
