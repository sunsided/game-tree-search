//! Property tests: generated tree shapes, table sizes and aspiration windows, shrunk to a
//! smallest failing case, and the laws every [`Score`] obeys.

use game_tree_search::{
    Algorithm, Bound, Deepening, Score, SearchResult, Searcher, SlotTable, Stats, TotalF64, Window,
    minimax,
};
use proptest::prelude::*;

use crate::games::RandomTree;

/// Every algorithm the [`Searcher`] runs.
const ALGORITHMS: [Algorithm; 3] = [Algorithm::AlphaBeta, Algorithm::Pvs, Algorithm::Mtdf];

/// Never stops a search early.
fn unlimited(_: &Stats) -> bool {
    false
}

/// Random trees of every shape the tests know: up to six moves a position, either move
/// order, either side satisficing or not, and with or without transpositions.
fn trees() -> impl Strategy<Value = RandomTree> {
    let satisfied = || prop::option::weighted(0.3, -100..=100i32);
    (
        0..=6u64,
        any::<bool>(),
        satisfied(),
        satisfied(),
        prop::option::of(1..=8u64),
    )
        .prop_map(|(max_branching, reversed, min, max, width)| RandomTree {
            max_branching,
            reversed,
            min_satisfied_below: min,
            max_satisfied_above: max,
            width,
            noisy: false,
        })
}

/// Whether `tree` ever stops at a good-enough line, which may leave a score inexact.
const fn satisfices(tree: &RandomTree) -> bool {
    tree.min_satisfied_below.is_some() || tree.max_satisfied_above.is_some()
}

/// A hook that asks for one fixed window on every attempt, never widening it.
struct Fixed(Window<i32>);

impl<M> Deepening<M, i32> for Fixed {
    fn proceed(&mut self, _: &SearchResult<M, i32>) -> bool {
        true
    }

    fn window(&mut self, _: &SearchResult<M, i32>, _: u32) -> Window<i32> {
        self.0
    }
}

/// Floats a [`TotalF64`] accepts: anything but NaN, the infinities and zeros included.
fn floats() -> impl Strategy<Value = TotalF64> {
    prop_oneof![
        any::<f64>().prop_filter_map("NaN", TotalF64::new),
        Just(TotalF64::new(f64::INFINITY).expect("not NaN")),
        Just(TotalF64::new(-0.0).expect("not NaN")),
    ]
}

/// The laws of a [`Score`] on `a` and `b`: negation is an involution that reverses the
/// order, and a step moves strictly, unless it saturates at the end of the range.
fn score_laws<S: Score + core::fmt::Debug>(a: S, b: S) -> Result<(), TestCaseError> {
    prop_assert_eq!(a.negate().negate(), a);
    prop_assert_eq!(a.cmp(&b).reverse(), a.negate().cmp(&b.negate()));
    prop_assert!(a.next_up() >= a && a.next_down() <= a);
    // A step stays put only at one end of the range, so the other step moves.
    prop_assert!(a.next_up() > a || a.next_down() < a);

    Ok(())
}

proptest! {
    /// Every algorithm, without a table and with a table so small that most positions
    /// share slots, returns a score whose bound admits minimax's, and the exact score when
    /// neither side satisfices. Deepening keeps the table between iterations, so stale and
    /// colliding entries are probed throughout.
    #[test]
    fn every_search_admits_the_minimax_score(
        tree in trees(),
        seed in any::<u64>(),
        depth in 0..=4u32,
        slots in 1..=16usize,
    ) {
        let value = minimax(&tree, &seed, depth).score;
        for algorithm in ALGORITHMS {
            let plain = Searcher::new().with_algorithm(algorithm).search(&tree, &seed, depth);
            let tabled = Searcher::with_table(SlotTable::new(slots))
                .with_algorithm(algorithm)
                .deepen(&tree, &seed, depth, unlimited);
            for result in [plain, tabled] {
                prop_assert!(result.bound.admits(result.score, value), "{algorithm:?}: {result:?}");
                if !satisfices(&tree) && result.depth == depth {
                    prop_assert_eq!((result.score, result.bound), (value, Bound::Exact));
                }
            }
        }
    }

    /// Any aspiration window, inverted, empty or far off the score included, still ends
    /// every iteration with minimax's exact score.
    #[test]
    fn any_window_ends_with_the_exact_score(
        max_branching in 0..=5u64,
        seed in any::<u64>(),
        depth in 1..=4u32,
        alpha in prop::option::of(-120..=120i32),
        beta in prop::option::of(-120..=120i32),
    ) {
        let tree = RandomTree::new(max_branching);
        for algorithm in [Algorithm::AlphaBeta, Algorithm::Pvs] {
            let result = Searcher::new()
                .with_algorithm(algorithm)
                .deepen_with(&tree, &seed, depth, unlimited, Fixed(Window { alpha, beta }));
            let value = minimax(&tree, &seed, result.depth).score;
            prop_assert_eq!((result.score, result.bound), (value, Bound::Exact), "{:?}", algorithm);
        }
    }

    /// Integer scores obey the [`Score`] laws over their whole range.
    #[test]
    fn integer_scores_obey_the_laws(a in any::<i32>(), b in any::<i32>(), c in any::<i8>(), d in any::<i8>()) {
        score_laws(a, b)?;
        score_laws(c, d)?;
    }

    /// Float scores obey the [`Score`] laws, zeros and infinities included.
    #[test]
    fn float_scores_obey_the_laws(a in floats(), b in floats()) {
        score_laws(a, b)?;
    }
}
