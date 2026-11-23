//! Single-threaded benchmarks: iterative deepening with every algorithm, with and without a
//! transposition table, and Monte Carlo tree search one leaf at a time and batched.

use std::hint::black_box;

use criterion::{BatchSize, Criterion};
use game_tree_search::{
    Algorithm, Gumbel, Mcts, Rollout, Searcher, SlotTable, Stats, TreeStats, Unbatched,
};

#[path = "../tests/it/games.rs"]
#[expect(
    dead_code,
    reason = "the test games are shared; the benches play only some of them"
)]
mod games;

use games::RandomTree;

/// The deepest iteration of the deepening benchmarks.
const DEPTH: u32 = 10;
/// Simulations per Monte Carlo search.
const SIMULATIONS: u64 = 20_000;
/// Root seeds, so one lucky tree does not decide a comparison.
const SEEDS: [u64; 4] = [1, 2, 3, 4];

/// A random tree of up to eight moves a position where every ply has 4096 positions, so move
/// orders transpose into each other as in real games.
const fn tree() -> RandomTree {
    let mut tree = RandomTree::new(8);
    tree.width = Some(4096);
    tree
}

/// Never stops a search early.
fn unlimited(_: &Stats) -> bool {
    false
}

/// Deepens each of [`SEEDS`] to [`DEPTH`] with each algorithm and a fresh [`Searcher`]
/// per seed, without a table and with a fresh [`SlotTable`] of 2^16 slots per seed, so no
/// seed is answered from another's entries.
fn bench_deepening(criterion: &mut Criterion) {
    let game = tree();
    let mut group = criterion.benchmark_group("deepen");
    for algorithm in [Algorithm::AlphaBeta, Algorithm::Pvs, Algorithm::Mtdf] {
        group.bench_function(format!("{algorithm:?}/no-table"), |bencher| {
            bencher.iter(|| {
                for seed in SEEDS {
                    black_box(
                        Searcher::new()
                            .with_algorithm(algorithm)
                            .deepen(&game, &seed, DEPTH, unlimited),
                    );
                }
            });
        });
        group.bench_function(format!("{algorithm:?}/slot-table"), |bencher| {
            bencher.iter_batched(
                || SEEDS.map(|_| SlotTable::new(1 << 16)),
                |tables| {
                    for (seed, table) in SEEDS.into_iter().zip(tables) {
                        black_box(
                            Searcher::with_table(table)
                                .with_algorithm(algorithm)
                                .deepen(&game, &seed, DEPTH, unlimited),
                        );
                    }
                },
                BatchSize::LargeInput,
            );
        });
    }

    group.finish();
}

/// Runs [`SIMULATIONS`] simulations with rollouts, one leaf at a time, in batches of eight
/// through [`Unbatched`], which shows what batching itself costs, and with Gumbel root
/// selection.
fn bench_mcts(criterion: &mut Criterion) {
    let game = tree();
    let budget = |stats: &TreeStats| stats.simulations >= SIMULATIONS;
    let mut group = criterion.benchmark_group("mcts");
    group.sample_size(20);
    group.bench_function("search", |bencher| {
        bencher.iter(|| Mcts::new().search(&game, &SEEDS[0], budget));
    });
    group.bench_function("search_batched/8", |bencher| {
        bencher.iter(|| {
            let mut evaluator = Unbatched::new(Rollout::default(), 7);
            Mcts::new().search_batched(&game, &SEEDS[0], 8, &mut evaluator, budget)
        });
    });
    group.bench_function("search_gumbel", |bencher| {
        let simulations = u32::try_from(SIMULATIONS).expect("a few thousand");
        bencher
            .iter(|| Mcts::new().search_gumbel(&game, &SEEDS[0], simulations, &Gumbel::default()));
    });

    group.finish();
}

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_deepening(&mut criterion);
    bench_mcts(&mut criterion);
    criterion.final_summary();
}
