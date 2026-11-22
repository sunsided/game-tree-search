//! Every algorithm on float scores and on integer scores at the ends of their range.

use game_tree_search::{
    Algorithm, Bound, Score, Searcher, Side, Stats, TotalF32, alpha_beta, minimax, mtdf, pvs,
};

use crate::algorithms::{ALGORITHMS, search};
use crate::games::{Nim, RandomTree, Rescored};

/// Seeds of the random trees the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// Checks that every algorithm returns the exact minimax score on `game`.
fn assert_exact<S: Score + core::fmt::Debug>(game: &Rescored<S>) {
    for algorithm in ALGORITHMS {
        for seed in SEEDS {
            for depth in 0..=5 {
                let result = search(algorithm, game, &seed, depth);
                let reference = minimax(game, &seed, depth).score;
                assert_eq!(
                    (result.score, result.bound),
                    (reference, Bound::Exact),
                    "seed {seed}"
                );
            }
        }
    }
}

/// Float scores, as autobattler uses, search exactly; null windows step through the float
/// order.
#[test]
fn float_scores_search_exactly() {
    let game = Rescored {
        tree: RandomTree::new(4),
        rescore: |score| {
            let score = f32::from(i16::try_from(score).expect("within i16"));
            TotalF32::new(score / 7.0).expect("finite")
        },
    };
    assert_exact(&game);
}

/// Scores at the ends of the integer range search exactly, where a null window cannot step
/// past the extreme.
#[test]
fn extreme_scores_search_exactly() {
    let game = Rescored {
        tree: RandomTree::new(4),
        rescore: |score| match score {
            ..-60 => i8::MIN,
            61.. => i8::MAX,
            _ => i8::try_from(score).expect("within i8"),
        },
    };
    assert_exact(&game);
}

/// Principal variation search scouts and re-searches on random trees; the node counts of
/// both algorithms under deepening are printed for comparison.
#[test]
fn pvs_researches_children_that_beat_the_scout() {
    let game = RandomTree::new(4);
    let mut totals = [Stats::default(); 2];
    for (total, algorithm) in totals.iter_mut().zip(ALGORITHMS) {
        let mut searcher = Searcher::new().with_algorithm(algorithm);
        for seed in SEEDS {
            let stats = searcher.deepen(&game, &seed, 7, |_: &Stats| false).stats;
            total.nodes += stats.nodes;
            total.researches += stats.researches;
        }
    }

    let [plain, pvs] = totals;
    println!(
        "deepening to 7: alpha-beta {} nodes, pvs {} nodes",
        plain.nodes, pvs.nodes
    );
    assert_eq!(plain.researches, 0);
    assert!(pvs.researches > 0);
    assert_eq!(ALGORITHMS[1], Algorithm::Pvs);
}

/// The one-call `pvs` and `mtdf` agree with alpha-beta, and a searcher reports its
/// algorithm.
#[test]
fn pvs_runs_principal_variation_search() {
    let root = (10, Side::Max);
    assert_eq!(
        pvs(&Nim, &root, 10).score,
        alpha_beta(&Nim, &root, 10).score
    );
    assert_eq!(
        mtdf(&Nim, &root, 10).score,
        alpha_beta(&Nim, &root, 10).score
    );

    let searcher = Searcher::<Nim>::new().with_algorithm(Algorithm::Pvs);
    assert_eq!(searcher.algorithm(), Algorithm::Pvs);
    assert_eq!(Searcher::<Nim>::default().algorithm(), Algorithm::AlphaBeta);
}

/// MTD(f) closes in over several passes and counts them, the other algorithms count none,
/// and a table makes the passes cheaper.
#[test]
fn mtdf_counts_its_passes() {
    use game_tree_search::SlotTable;

    let game = RandomTree {
        width: Some(3),
        ..RandomTree::new(4)
    };
    let mut plain = Searcher::new().with_algorithm(Algorithm::Mtdf);
    let mut tabled = Searcher::with_table(SlotTable::new(1 << 12)).with_algorithm(Algorithm::Mtdf);
    let (mut passes, mut plain_nodes, mut tabled_nodes) = (0, 0, 0);
    for seed in SEEDS {
        let result = plain.deepen(&game, &seed, 6, |_: &Stats| false);
        passes += result.stats.passes;
        plain_nodes += result.stats.nodes;
        tabled_nodes += tabled
            .deepen(&game, &seed, 6, |_: &Stats| false)
            .stats
            .nodes;
        assert_eq!(
            search(Algorithm::AlphaBeta, &game, &seed, 5).stats.passes,
            0
        );
    }

    println!(
        "mtdf deepening to 6: {plain_nodes} nodes, {tabled_nodes} with a table, {passes} passes"
    );
    assert!(passes > 2 * u64::try_from(SEEDS.count()).expect("small"));
    assert!(tabled_nodes < plain_nodes);
}
