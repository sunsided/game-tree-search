//! Parallel benchmarks (feature `parallel`): Lazy SMP over a [`SharedTable`] and over a
//! [`PackedTable`], and root- and tree-parallel Monte Carlo tree search, each at one to eight
//! threads.
//!
//! Single-threaded baselines sit next to them: a search on a [`SlotTable`] against the same
//! search on an uncontended [`SharedTable`] or [`PackedTable`] prices their synchronisation.
//! The MCTS groups compare independent trees (`search_parallel`) with one shared tree
//! (`search_shared`) at cheap and at network-like evaluation costs. The deepening benchmarks
//! play [`Narrow`] trees, whose `u8` moves pack.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion};
use game_tree_search::{
    Algorithm, Evaluation, Evaluator, Game, Hashed, Mcts, PackedTable, Random, Rewarded, Searcher,
    SharedTable, Side, SlotTable, Stats, Table, TreeStats, deepen_shared,
};

#[path = "../tests/it/games.rs"]
#[expect(
    dead_code,
    reason = "the test games are shared; the benches play only some of them"
)]
mod games;

use games::{Narrow, RandomTree};

/// The deepest iteration of the deepening benchmarks.
const DEPTH: u32 = 13;
/// The deepest iteration of the Lazy SMP benchmark on [`Costly`] nodes.
const COSTLY_DEPTH: u32 = 11;
/// Slots in every table, as a power of two.
const SLOTS: usize = 1 << 16;
/// Simulations per Monte Carlo search, over all threads.
const SIMULATIONS: u64 = 40_000;
/// Simulations per Monte Carlo search with [`Heavy`] evaluations, over all threads.
const HEAVY_SIMULATIONS: u64 = 2_000;
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

/// One thread deepening on a [`SlotTable`], and on a [`SharedTable`] and a [`PackedTable`]
/// nobody else touches: the differences are what the locks and the packing cost.
fn bench_table_locks(criterion: &mut Criterion) {
    let game = Narrow(tree());
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
    group.bench_function("packed", |bencher| {
        bencher.iter_batched(
            || PackedTable::new(SLOTS),
            |table| Searcher::with_table(&table).deepen(&game, &ROOT, DEPTH, unlimited),
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Lazy SMP at each thread count on both shared tables, fresh per search: on the plain
/// tree, whose nodes cost a few dozen nanoseconds, and on [`Costly`] nodes, closer to a
/// real game's.
fn bench_lazy_smp(criterion: &mut Criterion) {
    let plain = Narrow(tree());
    let costly = Costly(Narrow(tree()));
    let shared = || SharedTable::new(SLOTS, 64);
    let packed = || PackedTable::new(SLOTS);
    lazy_smp_group(criterion, "lazy_smp/shared", &plain, DEPTH, shared);
    lazy_smp_group(criterion, "lazy_smp/packed", &plain, DEPTH, packed);
    lazy_smp_group(
        criterion,
        "lazy_smp_costly/shared",
        &costly,
        COSTLY_DEPTH,
        shared,
    );
    lazy_smp_group(
        criterion,
        "lazy_smp_costly/packed",
        &costly,
        COSTLY_DEPTH,
        packed,
    );
}

/// The benchmark group `name`: Lazy SMP on `game` to `depth` at each of [`THREADS`], on a
/// table from `table` per search.
fn lazy_smp_group<G, T>(
    criterion: &mut Criterion,
    name: &str,
    game: &G,
    depth: u32,
    table: impl Fn() -> T,
) where
    G: Hashed<Position = u64, Move = u8, Score = i32, Key = u64> + Sync,
    for<'a> &'a T: Table<u64, u8, i32> + Send + Sync,
{
    let mut group = criterion.benchmark_group(name);
    for threads in THREADS {
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter_batched(
                    &table,
                    |table| {
                        deepen_shared(
                            game,
                            &ROOT,
                            depth,
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

/// The tree with an evaluation that costs about a microsecond, as a real game's often does,
/// so a node's own work outweighs the table's synchronisation.
struct Costly(Narrow);

/// Rounds of busy work in each [`Costly`] evaluation.
const EVALUATION_ROUNDS: u64 = 400;

impl Game for Costly {
    type Position = u64;
    type Move = u8;
    type Score = i32;

    fn side_to_move(&self, position: &u64) -> Side {
        self.0.side_to_move(position)
    }

    fn moves(&self, position: &u64, moves: &mut Vec<u8>) {
        self.0.moves(position, moves);
    }

    fn play(&self, position: &u64, mv: &u8) -> u64 {
        self.0.play(position, mv)
    }

    fn evaluate(&self, position: &u64) -> i32 {
        let busy = (0..EVALUATION_ROUNDS).fold(*position, |acc, round| {
            black_box(acc.rotate_left(7) ^ round).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        });
        black_box(busy);

        self.0.evaluate(position)
    }
}

impl Hashed for Costly {
    type Key = u64;

    fn key(&self, position: &u64) -> u64 {
        self.0.key(position)
    }
}

/// An evaluation that burns about 30 microseconds before it reads the tree's score, as a
/// neural network's might.
#[derive(Clone, Copy)]
struct Heavy;

/// Rounds of busy work in each [`Heavy`] evaluation.
const HEAVY_ROUNDS: u64 = 40_000;

impl<G: Rewarded<Position = u64>> Evaluator<G> for Heavy {
    fn evaluate<R: Random>(&mut self, game: &G, position: &u64, _: &mut R) -> Evaluation {
        let busy = (0..HEAVY_ROUNDS).fold(*position, |acc, round| {
            black_box(acc.rotate_left(7) ^ round).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        });
        black_box(busy);

        Evaluation::new(game.reward(game.evaluate(position)))
    }
}

/// Root-parallel and tree-parallel Monte Carlo tree search at each thread count, with the
/// same total simulations: with rollouts, whose leaves cost about a microsecond, and with
/// [`Heavy`] evaluations of about 30 microseconds, where the shared tree's lock stops
/// mattering.
fn bench_mcts_parallel(criterion: &mut Criterion) {
    let plain = tree();
    let simulations = SIMULATIONS;
    mcts_group(
        criterion,
        "mcts_parallel/root",
        simulations,
        |threads, budget| Mcts::new().search_parallel(&plain, &ROOT, threads, budget),
    );
    mcts_group(
        criterion,
        "mcts_parallel/shared",
        simulations,
        |threads, budget| Mcts::new().search_shared(&plain, &ROOT, threads, budget),
    );
    mcts_group(
        criterion,
        "mcts_heavy/root",
        HEAVY_SIMULATIONS,
        |threads, budget| {
            Mcts::new()
                .with_evaluator(Heavy)
                .search_parallel(&plain, &ROOT, threads, budget)
        },
    );
    mcts_group(
        criterion,
        "mcts_heavy/shared",
        HEAVY_SIMULATIONS,
        |threads, budget| {
            Mcts::new()
                .with_evaluator(Heavy)
                .search_shared(&plain, &ROOT, threads, budget)
        },
    );
}

/// The benchmark group `name`: `search` at each of [`THREADS`] with `total` simulations in
/// all, so root parallelism gives each thread its share and tree parallelism counts them on
/// the one tree.
fn mcts_group<R>(
    criterion: &mut Criterion,
    name: &str,
    total: u64,
    search: impl Fn(usize, Budget) -> R,
) {
    let mut group = criterion.benchmark_group(name);
    group.sample_size(20);
    let shared = name.ends_with("shared");
    for threads in THREADS {
        let count = u64::try_from(threads).expect("a few threads");
        let limit = if shared { total } else { total / count };
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter(|| search(threads, Budget(limit)));
            },
        );
    }

    group.finish();
}

/// A budget of a number of simulations.
#[derive(Clone, Copy)]
struct Budget(u64);

impl game_tree_search::Budget<TreeStats> for Budget {
    fn exhausted(&mut self, stats: &TreeStats) -> bool {
        stats.simulations >= self.0
    }
}

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_table_locks(&mut criterion);
    bench_lazy_smp(&mut criterion);
    bench_mcts_parallel(&mut criterion);
    criterion.final_summary();
}
