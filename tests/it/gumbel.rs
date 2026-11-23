//! Gumbel root selection: the Gumbel-max draw, the halving schedule, policy improvement,
//! symmetry between the sides, and how it treats kept trees.

use game_tree_search::{
    Evaluation, Evaluator, Game, Gumbel, Mcts, Random, Rewarded, Rollout, Side, SplitMix64,
    StaticValue, TreeStats,
};

use crate::games::RandomTree;

/// A one-move game: Max picks an arm, and the game ends with that arm's score. The root
/// scores `500`, an even prospect.
struct Bandit(Vec<i32>);

impl Game for Bandit {
    /// `None` before the pick, the arm after it.
    type Position = Option<usize>;
    /// The arm.
    type Move = usize;
    /// The arm's score, `0..=1000`.
    type Score = i32;

    fn side_to_move(&self, _: &Option<usize>) -> Side {
        Side::Max
    }

    fn moves(&self, position: &Option<usize>, moves: &mut Vec<usize>) {
        if position.is_none() {
            moves.extend(0..self.0.len());
        }
    }

    fn play(&self, _: &Option<usize>, &arm: &usize) -> Option<usize> {
        Some(arm)
    }

    fn evaluate(&self, position: &Option<usize>) -> i32 {
        position.map_or(500, |arm| self.0[arm])
    }
}

impl Rewarded for Bandit {
    /// Scores `0..=1000` mapped onto `[0, 1]`.
    fn reward(&self, score: i32) -> f64 {
        f64::from(score) / 1000.0
    }
}

/// An evaluator with fixed move priors and a neutral value.
#[derive(Clone)]
struct Priors(Vec<f64>);

impl<G: Rewarded> Evaluator<G> for Priors {
    fn evaluate<R: Random>(&mut self, _: &G, _: &G::Position, _: &mut R) -> Evaluation {
        Evaluation::new(0.5)
    }

    fn priors(&mut self, _: &G, _: &G::Position, moves: &[G::Move], priors: &mut Vec<f64>) {
        priors.extend(self.0.iter().take(moves.len()));
    }
}

/// Without simulations, the chosen move is the Gumbel-max draw from the priors: each move
/// is chosen about as often as its prior says.
#[test]
fn without_simulations_moves_follow_their_priors() {
    let priors = [0.6, 0.3, 0.1];
    let game = Bandit(vec![0, 0, 0]);
    let mut counts = [0u32; 3];
    let draws = 4000;
    for seed in 0..draws {
        let mut mcts = Mcts::new()
            .with_evaluator(Priors(priors.to_vec()))
            .with_rng(SplitMix64::new(seed));
        let result = mcts.search_gumbel(&game, &None, 0, &Gumbel::default());
        counts[result.best.expect("three arms")] += 1;
    }

    for (count, prior) in counts.iter().zip(priors) {
        let share = f64::from(*count) / f64::from(u32::try_from(draws).expect("small"));
        assert!((share - prior).abs() < 0.03, "{counts:?}");
    }
}

/// Eight moves and 24 simulations follow the halving schedule: every move once, four of
/// them twice more, two of them four times more.
#[test]
fn simulations_follow_the_halving_schedule() {
    let game = Bandit((0..8).map(|arm| 100 * arm).collect());
    let gumbel = Gumbel {
        considered: 8,
        ..Gumbel::default()
    };
    let result = Mcts::new()
        .with_evaluator(StaticValue)
        .search_gumbel(&game, &None, 24, &gumbel);
    let mut visits: Vec<u32> = result.root.iter().map(|root| root.visits).collect();
    visits.sort_unstable_by(|a, b| b.cmp(a));
    assert_eq!(visits, [7, 7, 3, 3, 1, 1, 1, 1]);
    assert_eq!(result.stats.simulations, 24);

    // The chosen move is one of the two most visited, and the line starts with it.
    let best = result.best.expect("eight arms");
    assert_eq!(result.root[best].visits, 7);
    assert_eq!(result.pv.first(), Some(&best));
}

/// With more moves than it considers, the search visits exactly the considered number of
/// moves: the halving schedule restricts the later rounds to moves already visited, as in
/// mctx.
#[test]
fn only_considered_moves_are_visited() {
    let game = Bandit((0..12).map(|arm| 80 * arm).collect());
    let gumbel = Gumbel {
        considered: 4,
        ..Gumbel::default()
    };
    for seed in 0..50 {
        let result = Mcts::new()
            .with_evaluator(StaticValue)
            .with_rng(SplitMix64::new(seed))
            .search_gumbel(&game, &None, 40, &gumbel);
        let visited = result.root.iter().filter(|root| root.visits > 0).count();
        assert_eq!(visited, 4, "seed {seed}");
    }
}

/// With exact values, the chosen move is better on average than a move drawn from the
/// uniform priors, and the improved policy is a distribution that favours better moves.
#[test]
fn the_search_improves_on_the_priors() {
    let scores = vec![100, 900, 300, 600, 200, 800, 400, 0];
    let game = Bandit(scores.clone());
    let uniform = f64::from(scores.iter().sum::<i32>()) / 8.0;
    let mut total = 0.0;
    for seed in 0..300 {
        let mut mcts = Mcts::new()
            .with_evaluator(StaticValue)
            .with_rng(SplitMix64::new(seed));
        let result = mcts.search_gumbel(&game, &None, 16, &Gumbel::default());
        total += f64::from(scores[result.best.expect("eight arms")]);

        let improved = result.improved.clone().expect("a Gumbel search sets it");
        assert!((improved.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(improved[1] > improved[0] && improved[5] > improved[7]);
        assert_eq!(result.policy(), improved);
    }

    assert!(total / 300.0 > uniform + 200.0, "{}", total / 300.0);
}

/// A [`RandomTree`] with the sides swapped and every score negated, so its rewards are
/// one minus the tree's.
struct Mirror(RandomTree);

impl Game for Mirror {
    /// The tree's positions.
    type Position = u64;
    /// The tree's moves.
    type Move = u64;
    /// The tree's scores, negated.
    type Score = i32;

    fn side_to_move(&self, position: &u64) -> Side {
        match self.0.side_to_move(position) {
            Side::Max => Side::Min,
            Side::Min => Side::Max,
        }
    }

    fn moves(&self, position: &u64, moves: &mut Vec<u64>) {
        self.0.moves(position, moves);
    }

    fn play(&self, position: &u64, mv: &u64) -> u64 {
        self.0.play(position, mv)
    }

    fn evaluate(&self, position: &u64) -> i32 {
        -self.0.evaluate(position)
    }
}

impl Rewarded for Mirror {
    /// The tree's mapping, so a negated score gives one minus the tree's reward.
    fn reward(&self, score: i32) -> f64 {
        f64::from(score + 100) / 200.0
    }
}

/// A searcher with short rollouts, its generator seeded with `seed`.
fn rollouts<G: Rewarded<Move = u64>>(seed: u64) -> Mcts<G, Rollout<u64>> {
    Mcts::new()
        .with_evaluator(Rollout::new(20))
        .with_rng(SplitMix64::new(seed))
}

/// A game and its mirror image, searched with the same seed, choose the same moves with
/// the same visits: the search treats both sides alike.
#[test]
fn both_sides_search_alike() {
    for seed in 0..40 {
        let gumbel = Gumbel::default();
        let plain = rollouts(seed).search_gumbel(&RandomTree::new(5), &seed, 64, &gumbel);
        let mirrored =
            rollouts(seed).search_gumbel(&Mirror(RandomTree::new(5)), &seed, 64, &gumbel);
        let visits = |result: &game_tree_search::MctsResult<u64>| {
            result
                .root
                .iter()
                .map(|root| root.visits)
                .collect::<Vec<_>>()
        };
        assert_eq!(plain.best, mirrored.best, "seed {seed}");
        assert_eq!(visits(&plain), visits(&mirrored), "seed {seed}");
        assert!(
            (plain.value + mirrored.value - 1.0).abs() < 1e-9,
            "seed {seed}"
        );
    }
}

/// Without Gumbel noise and with a deterministic evaluator, the seed changes nothing.
#[test]
fn without_noise_the_search_is_deterministic() {
    let gumbel = Gumbel {
        scale: 0.0,
        ..Gumbel::default()
    };
    let search = |seed| {
        Mcts::new()
            .with_evaluator(StaticValue)
            .with_rng(SplitMix64::new(seed))
            .search_gumbel(&RandomTree::new(4), &3, 50, &gumbel)
    };

    assert_eq!(search(1), search(2));
}

/// The root's own evaluation enters the value of moves not visited yet. After one visit to
/// an arm worth `0.6`, a root that evaluates to a sure win makes the unvisited arm look
/// better, `(1 + 0.6) / 2`; an even root would make it look worse, `(0.5 + 0.6) / 2`.
#[test]
fn unvisited_moves_mix_in_the_roots_own_value() {
    let game = Bandit(vec![600, 0]);
    let gumbel = Gumbel {
        scale: 0.0,
        ..Gumbel::default()
    };
    let result = Mcts::new()
        .with_evaluator(WinningRoot)
        .search_gumbel(&game, &None, 1, &gumbel);
    assert_eq!(result.root[0].visits, 1);
    let improved = result.improved.expect("a Gumbel search sets it");
    assert!(improved[1] > improved[0], "{improved:?}");
}

/// Evaluates the root of a [`Bandit`] as a sure win and an arm by its score.
struct WinningRoot;

impl Evaluator<Bandit> for WinningRoot {
    fn evaluate<R: Random>(&mut self, game: &Bandit, arm: &Option<usize>, _: &mut R) -> Evaluation {
        Evaluation::new(arm.map_or(1.0, |arm| game.reward(game.0[arm])))
    }
}

/// A Gumbel search starts a new tree even after [`Mcts::advance`] kept one, and leaves
/// nothing kept for the next search.
#[test]
fn gumbel_searches_start_new_trees() {
    let game = RandomTree::new(4);
    let mut mcts = Mcts::new();
    let budget = |stats: &TreeStats| stats.simulations >= 50;
    let _ = mcts.search(&game, &7, budget);
    assert!(mcts.advance(&[]));

    let gumbel = mcts.search_gumbel(&game, &7, 30, &Gumbel::default());
    let visits: u32 = gumbel.root.iter().map(|root| root.visits).sum();
    assert_eq!(visits, 30);

    let plain = mcts.search(&game, &7, budget);
    let visits: u32 = plain.root.iter().map(|root| root.visits).sum();
    assert_eq!(visits, 50);
}

/// Invalid settings are refused.
#[test]
#[should_panic(expected = "finite and non-negative")]
fn negative_scales_panic() {
    let gumbel = Gumbel {
        scale: -1.0,
        ..Gumbel::default()
    };
    let _ = Mcts::new().search_gumbel(&RandomTree::new(2), &1, 4, &gumbel);
}

/// A finished root returns at once, with nothing to choose.
#[test]
fn a_finished_root_has_no_move() {
    let result = Mcts::new().search_gumbel(&Bandit(vec![1, 2]), &Some(0), 10, &Gumbel::default());
    assert_eq!((result.best, result.improved), (None, None));
}
