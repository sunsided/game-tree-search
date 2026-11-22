//! Monte Carlo tree search: solved games, visit invariants, reproducibility and its plug
//! points.

use game_tree_search::{
    Evaluation, Evaluator, Game, Mcts, NodeStats, Puct, Random, Rewarded, Selection, Side,
    SplitMix64, StaticValue, TreeStats, minimax,
};

use std::cell::Cell;
use std::rc::Rc;

use crate::games::{Nim, RandomTree, Shallow};

/// A budget of `n` simulations.
fn simulations(n: u64) -> impl FnMut(&TreeStats) -> bool {
    move |stats: &TreeStats| stats.simulations >= n
}

/// On Nim, MCTS takes the winning move from every winning pile, for either side.
#[test]
fn mcts_solves_nim() {
    for side in [Side::Max, Side::Min] {
        let mut mcts = Mcts::new();
        for stones in (1..=10).filter(|stones| stones % 3 != 0) {
            let result = mcts.search(&Nim, &(stones, side), simulations(3000));
            assert_eq!(result.best, Some(stones % 3), "{stones} stones, {side:?}");
        }
    }
}

/// Every simulation passes through exactly one root move, and the line starts with the best.
#[test]
fn root_visits_add_up_to_the_simulations() {
    let game = RandomTree::new(4);
    let mut mcts = Mcts::new();
    for seed in 0..50 {
        let result = mcts.search(&game, &seed, simulations(200));
        assert_eq!(result.stats.simulations, 200);
        if result.root.is_empty() {
            assert_eq!(result.best, None);
            continue;
        }

        assert_eq!(
            result
                .root
                .iter()
                .map(|root| u64::from(root.visits))
                .sum::<u64>(),
            200
        );
        assert_eq!(result.pv.first(), result.best.as_ref(), "seed {seed}");
        let most = result.root.iter().map(|root| root.visits).max();
        let best = result
            .root
            .iter()
            .find(|root| Some(&root.mv) == result.best.as_ref());
        assert_eq!(best.map(|root| root.visits), most, "seed {seed}");
    }
}

/// Without simulations, the root's moves are listed unvisited and the first is chosen; a
/// finished game has no move.
#[test]
fn empty_budgets_and_finished_games() {
    let mut mcts = Mcts::new();
    let none = mcts.search(&Nim, &(5, Side::Max), simulations(0));
    assert_eq!((none.best, none.root.len(), none.pv.len()), (Some(1), 2, 0));
    assert!(
        none.root
            .iter()
            .all(|root| root.visits == 0 && (root.mean() - 0.5).abs() < 1e-12)
    );

    let finished = mcts.search(&Nim, &(0, Side::Max), simulations(10));
    assert_eq!((finished.best, finished.root.len()), (None, 0));
    assert_eq!(finished.stats.simulations, 10);
}

/// The same seed reproduces the same sequence of searches, reused searcher or not.
#[test]
fn seeds_reproduce_searches() {
    let game = RandomTree::new(4);
    let (mut a, mut b) = (Mcts::new(), Mcts::new());
    for seed in 0..20 {
        assert_eq!(
            a.search(&game, &seed, simulations(100)),
            b.search(&game, &seed, simulations(100)),
            "seed {seed}"
        );
    }
}

/// On small random trees with random sides to move, many simulations find the minimax move
/// wherever it beats every other root move by a clear margin.
#[test]
fn mcts_finds_clearly_best_moves() {
    let game = Shallow {
        tree: RandomTree {
            width: Some(1 << 20),
            ..RandomTree::new(3)
        },
        levels: 3,
    };
    let mut mcts = Mcts::new().with_evaluator(StaticValue);
    let mut checked = 0;
    for seed in 0..100 {
        let reference = minimax(&game, &seed, 3);
        let Some(best) = reference.best() else {
            continue;
        };
        let side = game.side_to_move(&seed);
        let mut values: Vec<i32> = reference.root_moves.iter().map(|root| root.score).collect();
        values.sort_unstable();
        let margin = if side == Side::Max {
            values.iter().rev().take(2).fold(0, |gap, &v| v - gap)
        } else {
            values.iter().take(2).fold(0, |gap, &v| gap - v)
        };
        if values.len() < 2 || margin.abs() < 40 {
            continue;
        }

        let result = mcts.search(&game, &seed, simulations(4000));
        assert_eq!(result.best.as_ref(), Some(best), "seed {seed}");
        checked += 1;
    }

    assert!(checked > 5, "{checked} positions checked");
}

/// Always the child with the most visits: a test-only selection rule.
struct MostVisited;

impl Selection for MostVisited {
    fn score(&self, _: Side, _: &NodeStats, child: &NodeStats) -> f64 {
        f64::from(child.visits)
    }
}

/// Counts its calls and values every leaf at one half: a test-only evaluator.
struct Counting {
    /// The leaves evaluated so far, shared with the test.
    calls: Rc<Cell<u64>>,
}

impl<G: Rewarded> Evaluator<G> for Counting {
    fn evaluate<R: Random>(&mut self, _: &G, _: &G::Position, _: &mut R) -> Evaluation {
        self.calls.set(self.calls.get() + 1);
        Evaluation::new(0.5)
    }
}

/// A generator that always returns zero: a test-only random source.
struct Zero;

impl Random for Zero {
    fn next_u64(&mut self) -> u64 {
        0
    }
}

/// Selection, evaluation and the random source are replaceable; every simulation that does
/// not end at a finished game evaluates one new leaf.
#[test]
fn plug_points_are_replaceable() {
    let calls = Rc::new(Cell::new(0));
    let mut mcts = Mcts::new()
        .with_selection(MostVisited)
        .with_evaluator(Counting {
            calls: Rc::clone(&calls),
        })
        .with_rng(Zero);
    let result = mcts.search(&Nim, &(9, Side::Max), simulations(50));
    assert_eq!(result.stats.simulations, 50);
    assert!(
        calls.get() > 0 && calls.get() <= 50,
        "{} evaluations",
        calls.get()
    );

    let mut rollouts = Mcts::default().with_rng(SplitMix64::new(9));
    let result = rollouts.search(&Nim, &(6, Side::Max), simulations(50));
    assert_eq!(result.stats.simulations, 50);
    let visited = result
        .root
        .iter()
        .find(|root| root.visits > 0)
        .expect("visited");
    assert!((visited.mean() - visited.value / f64::from(visited.visits)).abs() < 1e-12);
}

/// A node budget bounds the tree's size.
#[test]
fn a_node_budget_bounds_the_tree() {
    let game = RandomTree::new(4);
    let mut mcts = Mcts::new();
    for seed in 0..20 {
        let budget = |stats: &TreeStats| stats.nodes >= 30 || stats.simulations >= 10_000;
        let result = mcts.search(&game, &seed, budget);
        assert!(result.stats.nodes < 30 + 4, "seed {seed}");
        assert!(result.stats.depth >= 1 || result.root.is_empty());
    }
}

/// Records the piles it is asked to value: a test-only evaluator for Nim.
struct Piles {
    /// Every pile evaluated, shared with the test.
    seen: Rc<Cell<Vec<u32>>>,
}

impl Evaluator<Nim> for Piles {
    fn evaluate<R: Random>(&mut self, _: &Nim, position: &(u32, Side), _: &mut R) -> Evaluation {
        let mut seen = self.seen.take();
        seen.push(position.0);
        self.seen.set(seen);
        Evaluation::new(0.5)
    }
}

/// A finished game is valued by its reward, never handed to the evaluator, even the first
/// time a simulation reaches it.
#[test]
fn evaluators_never_see_finished_games() {
    let seen = Rc::new(Cell::new(Vec::new()));
    let mut mcts = Mcts::new().with_evaluator(Piles {
        seen: Rc::clone(&seen),
    });
    let result = mcts.search(&Nim, &(4, Side::Max), simulations(200));
    let seen = seen.take();
    assert!(!seen.is_empty() && !seen.contains(&0), "{seen:?}");
    assert_eq!(result.best, Some(1));
}

/// A `rand_core` generator drives the rollouts through `RandCore`, reproducibly.
#[cfg(feature = "rand_core")]
#[test]
fn rand_core_generators_drive_rollouts() {
    use core::convert::Infallible;

    use game_tree_search::RandCore;

    /// A counter as a `rand_core` generator: a test-only random source.
    struct Counter(u64);

    impl rand_core::TryRng for Counter {
        type Error = Infallible;

        fn try_next_u32(&mut self) -> Result<u32, Infallible> {
            self.try_next_u64()
                .map(|bits| u32::try_from(bits >> 32).expect("32 bits"))
        }

        fn try_next_u64(&mut self) -> Result<u64, Infallible> {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            Ok(self.0 ^ (self.0 >> 29))
        }

        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
            dst.fill(0);
            Ok(())
        }
    }

    let search = || {
        Mcts::new()
            .with_rng(RandCore(Counter(5)))
            .search(&Nim, &(7, Side::Max), simulations(2000))
    };
    let result = search();
    assert_eq!(result.best, Some(1));
    assert_eq!(result, search());
}

/// Values leaves by the game's evaluation and puts most of the prior on the move with
/// index `favourite` among each position's moves: a test-only policy.
#[derive(Clone)]
struct Hinted {
    /// The favoured move index; positions with fewer moves favour none.
    favourite: usize,
}

impl<G: Rewarded> Evaluator<G> for Hinted {
    fn evaluate<R: Random>(&mut self, game: &G, position: &G::Position, _: &mut R) -> Evaluation {
        Evaluation::new(game.reward(game.evaluate(position)))
    }

    fn priors(&mut self, _: &G, _: &G::Position, moves: &[G::Move], priors: &mut Vec<f64>) {
        priors
            .extend((0..moves.len()).map(|index| if index == self.favourite { 9.0 } else { 1.0 }));
    }
}

/// Scores a child by its prior alone, and scores unvisited children too: a test-only rule.
struct PriorOnly;

impl Selection for PriorOnly {
    fn score(&self, _: Side, _: &NodeStats, child: &NodeStats) -> f64 {
        child.prior
    }

    fn unvisited_first(&self) -> bool {
        false
    }
}

/// A rule that reads the priors and scores unvisited children follows the favoured move
/// only.
#[test]
fn priors_reach_the_selection_rule() {
    let mut mcts = Mcts::new()
        .with_evaluator(Hinted { favourite: 1 })
        .with_selection(PriorOnly);
    let result = mcts.search(&Nim, &(9, Side::Max), simulations(30));
    let visits: Vec<u32> = result.root.iter().map(|root| root.visits).collect();
    assert_eq!(visits, [0, 30]);
}

/// Under PUCT, priors steer the visits towards the favoured move, which UCT ignores.
#[test]
fn puct_follows_the_priors() {
    let game = RandomTree::new(6);
    let (mut puct_share, mut uct_share, mut roots) = (0.0, 0.0, 0);
    for seed in 0..100 {
        let mut moves = Vec::new();
        game.moves(&seed, &mut moves);
        if moves.len() < 3 {
            continue;
        }

        let share = |result: game_tree_search::MctsResult<u64>| {
            f64::from(result.root[1].visits)
                / f64::from(u32::try_from(result.stats.simulations).expect("small"))
        };
        let hinted = Hinted { favourite: 1 };
        let mut puct = Mcts::new()
            .with_evaluator(hinted.clone())
            .with_selection(Puct::default());
        let mut uct = Mcts::new().with_evaluator(hinted);
        puct_share += share(puct.search(&game, &seed, simulations(100)));
        uct_share += share(uct.search(&game, &seed, simulations(100)));
        roots += 1;
    }

    let roots = f64::from(roots);
    assert!(puct_share / roots > 0.5, "PUCT {}", puct_share / roots);
    assert!(
        puct_share > uct_share,
        "PUCT {puct_share} vs UCT {uct_share}"
    );
}

/// With uniform priors, PUCT still finds the winning Nim moves.
#[test]
fn puct_with_uniform_priors_solves_nim() {
    let mut mcts = Mcts::new().with_selection(Puct::default());
    for stones in (1..=10).filter(|stones| stones % 3 != 0) {
        let result = mcts.search(&Nim, &(stones, Side::Max), simulations(3000));
        assert_eq!(result.best, Some(stones % 3), "{stones} stones");
    }
}

/// The visits of all root moves together.
fn root_visits(result: &game_tree_search::MctsResult<u32>) -> u32 {
    result.root.iter().map(|root| root.visits).sum()
}

/// After `advance`, the next search continues from the kept subtree: its root moves start
/// with the visits the child had below it, and simulations count from zero.
#[test]
fn advancing_keeps_the_subtree() {
    let mut mcts = Mcts::new();
    let first = mcts.search(&Nim, &(10, Side::Max), simulations(500));
    let child = first.root.iter().find(|root| root.mv == 1).expect("a move");

    assert!(mcts.advance(&[1]));
    let kept = mcts.search(&Nim, &(9, Side::Min), simulations(0));
    assert_eq!(root_visits(&kept), child.visits - 1);

    assert!(mcts.advance(&[]));
    let more = mcts.search(&Nim, &(9, Side::Min), simulations(200));
    assert_eq!(more.stats.simulations, 200);
    assert_eq!(root_visits(&more), child.visits - 1 + 200);

    let fresh = mcts.search(&Nim, &(9, Side::Min), simulations(50));
    assert_eq!(root_visits(&fresh), 50);
}

/// Advancing by a move not in the tree, into a never visited position, or before any
/// search keeps nothing, and the next search starts a new tree.
#[test]
fn advancing_off_the_tree_starts_over() {
    let mut mcts = Mcts::new();
    assert!(!mcts.advance(&[1]));
    assert!(!mcts.advance(&[]));

    let _ = mcts.search(&Nim, &(10, Side::Max), simulations(1));
    assert!(!mcts.advance(&[2]));
    let after = mcts.search(&Nim, &(8, Side::Min), simulations(30));
    assert_eq!(root_visits(&after), 30);

    let _ = mcts.search(&Nim, &(10, Side::Max), simulations(100));
    assert!(!mcts.advance(&[3]));
}

/// Playing a whole game with tree reuse keeps the tree from growing past one search's worth
/// of nodes, and every reused search still finds the winning move.
#[test]
fn tree_reuse_bounds_memory_over_a_game() {
    let mut mcts = Mcts::new();
    let mut position = (10, Side::Max);
    let mut largest = 0;
    while position.0 > 0 {
        let result = mcts.search(&Nim, &position, simulations(3000));
        let best = result.best.expect("a move");
        if position.0 % 3 != 0 {
            assert_eq!(best, position.0 % 3, "{} stones", position.0);
        }

        largest = largest.max(result.stats.nodes);
        position = Nim.play(&position, &best);
        let _ = mcts.advance(&[best]);
    }

    assert!(largest < 3000 * 3, "{largest} nodes");
}

/// A search of a position that does not fit the kept subtree, such as the root of a new
/// game, starts a new tree instead of continuing the old one.
#[test]
fn a_kept_tree_is_dropped_for_another_position() {
    for (root, moves) in [((9, Side::Max), 2), ((1, Side::Min), 1)] {
        let mut mcts = Mcts::new();
        let _ = mcts.search(&Nim, &(10, Side::Max), simulations(500));
        assert!(mcts.advance(&[1]));

        let other = mcts.search(&Nim, &root, simulations(40));
        assert_eq!(root_visits(&other), 40, "{root:?}");
        assert_eq!(other.root.len(), moves, "{root:?}");
    }
}

/// Root noise with a zero share changes nothing; with a share, it is reproducible from the
/// seed and changes where PUCT spends its visits.
#[test]
fn root_noise_is_reproducible_and_steers_puct() {
    use game_tree_search::RootNoise;

    let game = RandomTree::new(6);
    let mut changed = 0;
    for seed in 0..50 {
        let search = |noise: Option<RootNoise>| {
            let mcts = Mcts::new()
                .with_evaluator(StaticValue)
                .with_selection(Puct::default());
            let mut mcts = match noise {
                Some(noise) => mcts.with_root_noise(noise),
                None => mcts,
            };
            mcts.search(&game, &seed, simulations(100))
        };
        let plain = search(None);
        let silent = search(Some(RootNoise {
            alpha: 0.3,
            fraction: 0.0,
        }));
        let noisy = search(Some(RootNoise {
            alpha: 0.3,
            fraction: 0.5,
        }));
        assert_eq!(plain.root, silent.root, "seed {seed}");
        assert_eq!(
            noisy,
            search(Some(RootNoise {
                alpha: 0.3,
                fraction: 0.5
            })),
            "seed {seed}"
        );
        changed += usize::from(noisy.root != plain.root);
    }

    assert!(changed > 0);
}

/// The root value is the root's mean reward: high where the side to move wins.
#[test]
fn the_root_value_reports_who_wins() {
    let mut mcts = Mcts::new();
    let won = mcts.search(&Nim, &(7, Side::Max), simulations(2000));
    let lost = mcts.search(&Nim, &(6, Side::Max), simulations(2000));
    assert!(
        won.value > 0.6 && lost.value < 0.4,
        "{} {}",
        won.value,
        lost.value
    );
    assert!((mcts.search(&Nim, &(7, Side::Max), simulations(0)).value - 0.5).abs() < 1e-12);
}
