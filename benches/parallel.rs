//! Parallel benchmarks (feature `parallel`): Lazy SMP over a [`SharedTable`] and
//! root-parallel Monte Carlo tree search, each at one to eight threads.
//!
//! Single-threaded baselines sit next to them: a search on a [`SlotTable`] against the same
//! search on an uncontended [`SharedTable`] prices the shard locks (issue #38), and the
//! thread counts of `search_parallel` show how far independent trees scale (issue #40).

use criterion::{BatchSize, BenchmarkId, Criterion};
use game_tree_search::{
    Algorithm, Mcts, Searcher, SharedTable, SlotTable, Stats, TreeStats, deepen_shared,
};

#[path = "../tests/it/games.rs"]
#[expect(
    dead_code,
    reason = "the test games are shared; the benches play only some of them"
)]
mod games;

use games::RandomTree;

/// The deepest iteration of the deepening benchmarks.
const DEPTH: u32 = 13;
/// Slots in every table, as a power of two.
const SLOTS: usize = 1 << 16;
/// Simulations per Monte Carlo search, over all threads.
const SIMULATIONS: u64 = 40_000;
/// Thread counts to compare.
const THREADS: [usize; 4] = [1, 2, 4, 8];
/// The root of every search.
const ROOT: u64 = 1;

/// A random tree of up to eight moves a position where every ply has 4096 positions, so
/// threads meet each other's positions in the shared table.
const fn tree() -> RandomTree {
    let mut tree = RandomTree::new(8);
    tree.width = Some(4096);
    tree
}

/// Never stops a search early.
fn unlimited(_: &Stats) -> bool {
    false
}

/// One thread deepening on a [`SlotTable`] and on a [`SharedTable`] nobody else touches:
/// the difference is what the shard locks cost.
fn bench_table_locks(criterion: &mut Criterion) {
    let game = tree();
    let mut group = criterion.benchmark_group("table");
    group.bench_function("slot", |bencher| {
        bencher.iter_batched(
            || SlotTable::new(SLOTS),
            |table| Searcher::with_table(table).deepen(&game, &ROOT, DEPTH, unlimited),
            BatchSize::LargeInput,
        );
    });
    group.bench_function("shared", |bencher| {
        bencher.iter_batched(
            || SharedTable::new(SLOTS, 64),
            |table| Searcher::with_table(&table).deepen(&game, &ROOT, DEPTH, unlimited),
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Lazy SMP at each thread count, on a fresh table per search.
fn bench_lazy_smp(criterion: &mut Criterion) {
    let game = tree();
    let mut group = criterion.benchmark_group("lazy_smp");
    for threads in THREADS {
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter_batched(
                    || SharedTable::new(SLOTS, 64),
                    |table| {
                        deepen_shared(
                            &game,
                            &ROOT,
                            DEPTH,
                            (threads, Algorithm::AlphaBeta),
                            &table,
                            unlimited,
                        )
                    },
                    BatchSize::LargeInput,
                );
            },
        );
    }

    group.finish();
}

/// Root-parallel Monte Carlo tree search at each thread count, the simulations split
/// evenly, so perfect scaling halves the time with every doubling.
fn bench_mcts_parallel(criterion: &mut Criterion) {
    let game = tree();
    let mut group = criterion.benchmark_group("mcts_parallel");
    group.sample_size(20);
    for threads in THREADS {
        let share = SIMULATIONS / u64::try_from(threads).expect("a few threads");
        let budget = move |stats: &TreeStats| stats.simulations >= share;
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter(|| Mcts::new().search_parallel(&game, &ROOT, threads, budget));
            },
        );
    }

    group.finish();
}

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_table_locks(&mut criterion);
    bench_lazy_smp(&mut criterion);
    bench_mcts_parallel(&mut criterion);
    criterion.final_summary();
}
