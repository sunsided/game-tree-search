//! Cross-checks of searches with a transposition table against plain minimax.

use game_tree_search::{
    Bound, SearchResult, Searcher, Side, SlotTable, Stats, Transpositions, alpha_beta, deepen,
    minimax,
};

use game_tree_search::Game;

use crate::games::{Nim, RandomTree};

/// Seeds of the random games the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// A searcher over random games with a slot table.
type TableSearcher = Searcher<RandomTree, Transpositions<SlotTable<u64, u64, i32>>>;

/// A searcher over random games with a table of `capacity` slots.
fn searcher(capacity: usize) -> TableSearcher {
    Searcher::with_table(SlotTable::new(capacity))
}

/// Random games where each ply has only three positions, so transpositions abound.
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

/// Checks what every result promises with a table: the root, every principal-variation step
/// present and every root move admit the minimax value of their position.
fn assert_admits(game: &RandomTree, seed: u64, depth: u32, result: &SearchResult<u64, i32>) {
    let value = minimax(game, &seed, depth).score;
    assert!(
        result.bound.admits(result.score, value),
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
        let value = minimax(game, &game.play(&seed, &root_move.mv), depth - 1).score;
        assert!(
            root_move.bound.admits(root_move.score, value),
            "seed {seed}"
        );
    }
}

/// With a table, alpha-beta still returns the exact minimax score, in either move order,
/// with one table reused across all roots and depths; the table is actually hit.
#[test]
fn a_reused_table_keeps_alpha_beta_exact() {
    for reversed in [false, true] {
        let game = RandomTree {
            reversed,
            ..narrow()
        };
        let mut searcher = searcher(1 << 12);
        let mut hits = 0;
        for seed in SEEDS {
            for depth in 0..=6 {
                let result = searcher.alpha_beta(&game, &seed, depth);
                assert_eq!(
                    result.score,
                    minimax(&game, &seed, depth).score,
                    "seed {seed}"
                );
                assert_eq!(result.bound, Bound::Exact, "seed {seed}, depth {depth}");
                assert_admits(&game, seed, depth, &result);
                hits += result.stats.table_hits;
            }
        }

        assert!(hits > 0);
    }
}

/// Transpositions within a single search are answered from a fresh table.
#[test]
fn a_single_search_hits_its_own_entries() {
    let game = narrow();
    let hits: u64 = SEEDS
        .map(|seed| {
            searcher(1 << 12)
                .alpha_beta(&game, &seed, 6)
                .stats
                .table_hits
        })
        .sum();
    assert!(hits > 0);
}

/// Tiny tables, where most entries are evicted, stay exact.
#[test]
fn tiny_tables_stay_exact() {
    let game = narrow();
    for capacity in [1, 4] {
        let mut searcher = searcher(capacity);
        for seed in SEEDS {
            let result = searcher.alpha_beta(&game, &seed, 5);
            assert_eq!(result.score, minimax(&game, &seed, 5).score, "seed {seed}");
        }
    }
}

/// Under two-sided satisficing, table results stay sound.
#[test]
fn satisficing_with_a_table_admits_the_minimax_value() {
    let game = RandomTree {
        min_satisfied_below: Some(0),
        max_satisfied_above: Some(-10),
        ..narrow()
    };
    let mut searcher = searcher(1 << 12);
    for seed in SEEDS {
        for depth in 1..=5 {
            let result = searcher.alpha_beta(&game, &seed, depth);
            assert_admits(&game, seed, depth, &result);
        }
    }
}

/// Deepening with a table reaches the same score as without, and reuses entries across
/// iterations and roots.
#[test]
fn deepening_with_a_table_matches_deepening_without() {
    let game = narrow();
    let mut searcher = searcher(1 << 12);
    let mut hits = 0;
    for seed in SEEDS {
        let remembered = searcher.deepen(&game, &seed, 6, unlimited);
        let plain = deepen(&game, &seed, 6, unlimited);
        assert_eq!(remembered.depth, plain.depth, "seed {seed}");
        assert_eq!(remembered.score, plain.score, "seed {seed}");
        hits += remembered.stats.table_hits;
    }

    assert!(hits > 0);
}

/// A search the budget stopped leaves nothing wrong in the table for the next search.
#[test]
fn a_table_survives_a_budget_abort() {
    let game = narrow();
    let mut searcher = searcher(1 << 12);
    for seed in SEEDS {
        let _ = searcher.deepen(&game, &seed, 8, |stats: &Stats| stats.nodes >= 50);
        let result = searcher.alpha_beta(&game, &seed, 6);
        assert_eq!(result.score, minimax(&game, &seed, 6).score, "seed {seed}");
    }
}

/// Clearing the table forgets every entry.
#[test]
fn clearing_the_table_forgets_entries() {
    let mut searcher = Searcher::with_table(SlotTable::new(64));
    let first = searcher.alpha_beta(&Nim, &(9, Side::Max), 9);
    searcher.table_mut().clear();
    let cleared = searcher.alpha_beta(&Nim, &(9, Side::Max), 9);
    let warm = searcher.alpha_beta(&Nim, &(9, Side::Max), 9);
    assert_eq!(first.stats, cleared.stats);
    assert!(warm.stats.nodes < cleared.stats.nodes);
    assert_eq!(searcher.table().capacity(), 64);
    assert_eq!(warm.score, alpha_beta(&Nim, &(9, Side::Max), 9).score);
}

/// The hash-map table keeps alpha-beta exact too, and a capped one stays exact once full.
#[cfg(feature = "hashbrown")]
#[test]
fn a_map_table_keeps_alpha_beta_exact() {
    use game_tree_search::MapTable;

    let game = narrow();
    for capacity in [1 << 12, 8] {
        let mut searcher = Searcher::with_table(MapTable::new(capacity));
        let mut hits = 0;
        for seed in SEEDS {
            let result = searcher.alpha_beta(&game, &seed, 5);
            assert_eq!(result.score, minimax(&game, &seed, 5).score, "seed {seed}");
            hits += result.stats.table_hits;
        }

        assert!(hits > 0);
        assert!(searcher.table().len() <= capacity);
    }
}
