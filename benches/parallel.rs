//! Parallel benchmarks (feature `parallel`): Lazy SMP over a [`SharedTable`] and
//! root-parallel Monte Carlo tree search, each at one to eight threads.
//!
//! Single-threaded baselines sit next to them: a search on a [`SlotTable`] against the same
//! search on an uncontended [`SharedTable`] prices the shard locks (issue #38), and the
//! thread counts of `search_parallel` show how far independent trees scale (issue #40).

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion};
use game_tree_search::{
    Algorithm, Game, Hashed, Mcts, Searcher, SharedTable, Side, SlotTable, Stats, TreeStats,
    deepen_shared,
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
/// The deepest iteration of the Lazy SMP benchmark on [`Costly`] nodes.
const COSTLY_DEPTH: u32 = 11;
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

/// Lazy SMP at each thread count, on a fresh table per search: on the plain tree, whose
/// nodes cost a few dozen nanoseconds, and on [`Costly`] nodes, closer to a real game's.
fn bench_lazy_smp(criterion: &mut Criterion) {
    let plain = tree();
    lazy_smp_group(criterion, "lazy_smp", &plain, DEPTH);
    let costly = Costly(tree());
    lazy_smp_group(criterion, "lazy_smp_costly", &costly, COSTLY_DEPTH);
}

/// The benchmark group `name`: Lazy SMP on `game` to `depth` at each of [`THREADS`].
fn lazy_smp_group<G>(criterion: &mut Criterion, name: &str, game: &G, depth: u32)
where
    G: Hashed<Position = u64, Move = u64, Score = i32, Key = u64> + Sync,
{
    let mut group = criterion.benchmark_group(name);
    for threads in THREADS {
        group.bench_with_input(
            BenchmarkId::from_parameter(threads),
            &threads,
            |bencher, &threads| {
                bencher.iter_batched(
                    || SharedTable::new(SLOTS, 64),
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
struct Costly(RandomTree);

/// Rounds of busy work in each [`Costly`] evaluation.
const EVALUATION_ROUNDS: u64 = 400;

impl Game for Costly {
    type Position = u64;
    type Move = u64;
    type Score = i32;

    fn side_to_move(&self, position: &u64) -> Side {
        self.0.side_to_move(position)
    }

    fn moves(&self, position: &u64, moves: &mut Vec<u64>) {
        self.0.moves(position, moves);
    }

    fn play(&self, position: &u64, mv: &u64) -> u64 {
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
