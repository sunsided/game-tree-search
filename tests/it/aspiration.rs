//! Cross-checks of deepening with aspiration windows against plain deepening and minimax.

use game_tree_search::{
    Algorithm, Bound, Deepening, Score, SearchResult, Searcher, SlotTable, Stats, TotalF32, Window,
    minimax,
};

use crate::algorithms::{ALGORITHMS, deepen_with_algorithm, search};
use crate::games::{RandomTree, Rescored};

/// Seeds of the random trees the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// A budget that never runs out.
fn unlimited(_: &Stats) -> bool {
    false
}

/// Aspiration windows `width` to either side of the last score, widened by `widen` on each
/// failed attempt.
struct Aspiration<S> {
    /// The first half-width.
    width: S,
    /// The half-width after one more failure.
    widen: fn(S) -> S,
    /// Moves a score by a signed half-width: `shift(score, width, up)`.
    shift: fn(S, S, bool) -> S,
}

impl<M, S: Score> Deepening<M, S> for Aspiration<S> {
    fn proceed(&mut self, _: &SearchResult<M, S>) -> bool {
        true
    }

    fn window(&mut self, previous: &SearchResult<M, S>, attempt: u32) -> Window<S> {
        let width = (0..attempt).fold(self.width, |width, _| (self.widen)(width));
        Window {
            alpha: Some((self.shift)(previous.score, width, false)),
            beta: Some((self.shift)(previous.score, width, true)),
        }
    }
}

/// Integer aspiration starting `width` points around the last score, doubling on failure.
fn integer(width: i32) -> Aspiration<i32> {
    Aspiration {
        width,
        widen: |width| width.saturating_mul(2),
        shift: |score, width, up| {
            if up {
                score.saturating_add(width)
            } else {
                score.saturating_sub(width)
            }
        },
    }
}

/// A hook that asks for the same window every time: never widens.
struct Stubborn(Window<i32>);

impl<M> Deepening<M, i32> for Stubborn {
    fn proceed(&mut self, _: &SearchResult<M, i32>) -> bool {
        true
    }

    fn window(&mut self, _: &SearchResult<M, i32>, _: u32) -> Window<i32> {
        self.0
    }
}

/// Aspirated deepening returns the score and exact bound of plain deepening, reaching its
/// depth unless proven sooner, for every algorithm; windows fail and are widened, except
/// for MTD(f), which ignores them.
#[test]
fn aspiration_matches_plain_deepening() {
    let game = RandomTree::new(4);
    for algorithm in ALGORITHMS {
        let mut searcher = Searcher::new().with_algorithm(algorithm);
        let mut widenings = 0;
        for seed in SEEDS {
            let aspirated = searcher.deepen_with(&game, &seed, 6, unlimited, integer(5));
            let plain = deepen_with_algorithm(algorithm, &game, &seed, 6, unlimited);
            // Narrow windows prune more, so an iteration can be proven sooner.
            let same_depth = aspirated.depth == plain.depth;
            assert!(same_depth || aspirated.proven, "{algorithm:?}, seed {seed}");
            assert_eq!(aspirated.score, plain.score, "{algorithm:?}, seed {seed}");
            assert_eq!(aspirated.bound, Bound::Exact, "{algorithm:?}, seed {seed}");
            widenings += aspirated.stats.widenings;
        }

        if algorithm == Algorithm::Mtdf {
            assert_eq!(widenings, 0);
        } else {
            assert!(widenings > 0, "{algorithm:?}");
        }
    }
}

/// A hook that never widens, or asks for an inverted window, still ends every iteration
/// with the exact score, at most a fixed number of attempts later.
#[test]
fn stubborn_and_inverted_windows_still_finish_exactly() {
    let game = RandomTree::new(4);
    let windows = [
        Window {
            alpha: Some(0),
            beta: Some(1),
        },
        Window {
            alpha: Some(10),
            beta: Some(-10),
        },
    ];
    for window in windows {
        for algorithm in [Algorithm::AlphaBeta, Algorithm::Pvs] {
            let mut searcher = Searcher::new().with_algorithm(algorithm);
            for seed in SEEDS {
                let result = searcher.deepen_with(&game, &seed, 5, unlimited, Stubborn(window));
                let reference = minimax(&game, &seed, result.depth).score;
                assert_eq!(
                    (result.score, result.bound),
                    (reference, Bound::Exact),
                    "seed {seed}"
                );
                assert!(
                    result.stats.widenings <= 8 * u64::from(result.depth),
                    "seed {seed}"
                );
            }
        }
    }
}

/// Aspiration under two-sided satisficing, where even a full window may be inexact, stays
/// sound and finishes.
#[test]
fn aspiration_with_satisficing_admits_the_minimax_value() {
    let game = RandomTree {
        min_satisfied_below: Some(0),
        max_satisfied_above: Some(-10),
        ..RandomTree::new(4)
    };
    for algorithm in ALGORITHMS {
        let mut searcher = Searcher::new().with_algorithm(algorithm);
        for seed in SEEDS {
            let result = searcher.deepen_with(&game, &seed, 5, unlimited, integer(3));
            let value = minimax(&game, &seed, result.depth).score;
            assert!(
                result.bound.admits(result.score, value),
                "{algorithm:?}, seed {seed}"
            );
        }
    }
}

/// Aspiration with a transposition table stays exact.
#[test]
fn aspiration_with_a_table_stays_exact() {
    let game = RandomTree {
        width: Some(3),
        ..RandomTree::new(4)
    };
    for algorithm in ALGORITHMS {
        let mut searcher = Searcher::with_table(SlotTable::new(1 << 12)).with_algorithm(algorithm);
        for seed in SEEDS {
            let result = searcher.deepen_with(&game, &seed, 6, unlimited, integer(2));
            let reference = search(Algorithm::AlphaBeta, &game, &seed, result.depth).score;
            assert_eq!(
                (result.score, result.bound),
                (reference, Bound::Exact),
                "seed {seed}"
            );
        }
    }
}

/// A budget that runs out during an aspiration attempt keeps the last finished iteration.
#[test]
fn a_budget_abort_mid_attempt_keeps_the_last_iteration() {
    let game = RandomTree::new(4);
    for algorithm in ALGORITHMS {
        let mut searcher = Searcher::new().with_algorithm(algorithm);
        for seed in SEEDS {
            let budget = |stats: &Stats| stats.nodes >= 150;
            let result = searcher.deepen_with(&game, &seed, 10, budget, integer(1));
            let reference = search(Algorithm::AlphaBeta, &game, &seed, result.depth).score;
            assert_eq!(result.score, reference, "{algorithm:?}, seed {seed}");
        }
    }
}

/// Float scores aspirate with a float half-width, the case that needs a caller-chosen
/// width; node counts with and without aspiration are printed for comparison.
#[test]
fn float_scores_aspirate() {
    let game = Rescored {
        tree: RandomTree::new(4),
        rescore: |score| {
            let score = f32::from(i16::try_from(score).expect("within i16"));
            TotalF32::new(score / 7.0).expect("finite")
        },
    };
    let aspiration = || Aspiration {
        width: TotalF32::new(1.0).expect("finite"),
        widen: |width| TotalF32::new(width.get() * 2.0).expect("finite"),
        shift: |score, width, up| {
            let shifted = if up {
                score.get() + width.get()
            } else {
                score.get() - width.get()
            };
            TotalF32::new(shifted).expect("finite")
        },
    };
    let (mut aspirated_nodes, mut plain_nodes) = (0, 0);
    for algorithm in [Algorithm::AlphaBeta, Algorithm::Pvs] {
        let mut searcher = Searcher::new().with_algorithm(algorithm);
        for seed in SEEDS {
            let aspirated = searcher.deepen_with(&game, &seed, 6, unlimited, aspiration());
            let plain = searcher.deepen(&game, &seed, 6, unlimited);
            assert_eq!(aspirated.score, plain.score, "seed {seed}");
            assert_eq!(aspirated.bound, Bound::Exact, "seed {seed}");
            aspirated_nodes += aspirated.stats.nodes;
            plain_nodes += plain.stats.nodes;
        }
    }

    println!("float deepening to 6: {aspirated_nodes} nodes aspirated, {plain_nodes} plain");
}
