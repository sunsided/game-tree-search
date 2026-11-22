//! Batched leaf evaluation in Monte Carlo tree search: batches of one equal plain searches,
//! budgets stay exact, and virtual loss spreads a batch.

use game_tree_search::{
    BatchEvaluator, BatchLeaf, Mcts, Puct, Rewarded, Side, StaticValue, TreeStats, Unbatched,
};

use crate::games::{Nim, RandomTree};

/// A budget of `n` simulations.
fn simulations(n: u64) -> impl FnMut(&TreeStats) -> bool {
    move |stats: &TreeStats| stats.simulations >= n
}

/// Values every leaf by the game's evaluation and favours the move at index `favourite`
/// with priors; records the batch sizes: a test-only batch evaluator.
struct Recording {
    /// The favoured move index, or `None` for no priors.
    favourite: Option<usize>,
    /// Whether to leave the values out.
    silent: bool,
    /// The size of every batch evaluated.
    sizes: Vec<usize>,
}

impl<G: Rewarded> BatchEvaluator<G> for Recording {
    fn evaluate_batch(
        &mut self,
        game: &G,
        leaves: &[BatchLeaf<'_, G::Position, G::Move>],
        values: &mut Vec<f64>,
        priors: &mut Vec<f64>,
    ) {
        self.sizes.push(leaves.len());
        if !self.silent {
            values.extend(
                leaves
                    .iter()
                    .map(|leaf| game.reward(game.evaluate(leaf.position))),
            );
        }

        if let Some(favourite) = self.favourite {
            for leaf in leaves {
                priors
                    .extend((0..leaf.moves.len()).map(|i| if i == favourite { 9.0 } else { 1.0 }));
            }
        }
    }
}

/// Batches of one leaf search exactly like a plain search with the same evaluator.
#[test]
fn batches_of_one_equal_plain_searches() {
    let game = RandomTree::new(4);
    for seed in 0..50 {
        for batch in [0, 1] {
            let mut evaluator = Unbatched::new(StaticValue, 0);
            let batched =
                Mcts::new().search_batched(&game, &seed, batch, &mut evaluator, simulations(200));
            let plain =
                Mcts::new()
                    .with_evaluator(StaticValue)
                    .search(&game, &seed, simulations(200));
            assert_eq!(batched, plain, "seed {seed}, batch {batch}");
        }
    }
}

/// A budget of `n` simulations runs exactly `n`, in batches of at most the batch size.
#[test]
fn budgets_stay_exact_in_batches() {
    let game = RandomTree::new(4);
    for seed in 0..50 {
        let mut evaluator = Recording {
            favourite: None,
            silent: false,
            sizes: Vec::new(),
        };
        let result = Mcts::new().search_batched(&game, &seed, 8, &mut evaluator, simulations(100));
        assert_eq!(result.stats.simulations, 100, "seed {seed}");
        if !result.root.is_empty() {
            assert_eq!(
                result
                    .root
                    .iter()
                    .map(|root| u64::from(root.visits))
                    .sum::<u64>(),
                100
            );
        }

        assert!(
            evaluator.sizes.iter().all(|&size| (1..=8).contains(&size)),
            "seed {seed}"
        );
    }
}

/// Virtual loss spreads one batch over several root moves even under PUCT, which, with
/// priors favouring one move, would otherwise send every simulation into it.
#[test]
fn virtual_loss_spreads_a_batch() {
    let game = RandomTree::new(6);
    let mut spread = 0;
    for seed in 0..50 {
        let mut evaluator = Recording {
            favourite: Some(0),
            silent: false,
            sizes: Vec::new(),
        };
        let mut mcts = Mcts::new().with_selection(Puct::default());
        let result = mcts.search_batched(&game, &seed, 6, &mut evaluator, simulations(6));
        spread += usize::from(result.root.iter().filter(|root| root.visits > 0).count() > 1);
    }

    assert!(spread > 0);
}

/// Priors returned by the batch steer PUCT towards the favoured move.
#[test]
fn batch_priors_steer_puct() {
    let game = RandomTree::new(6);
    let (mut favoured, mut total) = (0, 0);
    for seed in 0..50 {
        let mut evaluator = Recording {
            favourite: Some(1),
            silent: false,
            sizes: Vec::new(),
        };
        let mut mcts = Mcts::new().with_selection(Puct::default());
        let result = mcts.search_batched(&game, &seed, 4, &mut evaluator, simulations(200));
        if result.root.len() >= 3 {
            favoured += result.root[1].visits;
            total += result.root.iter().map(|root| root.visits).sum::<u32>();
        }
    }

    assert!(
        f64::from(favoured) > 0.4 * f64::from(total),
        "{favoured} of {total}"
    );
}

/// Missing values count as one half; the search still finishes its budget.
#[test]
fn missing_values_count_as_one_half() {
    let mut evaluator = Recording {
        favourite: None,
        silent: true,
        sizes: Vec::new(),
    };
    let result =
        Mcts::new().search_batched(&Nim, &(20, Side::Max), 4, &mut evaluator, simulations(40));
    assert_eq!(result.stats.simulations, 40);
    assert!(
        result
            .root
            .iter()
            .all(|root| (root.mean() - 0.5).abs() < 1e-9)
    );
}

/// Rollouts in batches of eight still solve Nim.
#[test]
fn batched_rollouts_solve_nim() {
    for stones in (1..=10).filter(|stones| stones % 3 != 0) {
        let mut evaluator = Unbatched::new(game_tree_search::Rollout::default(), stones.into());
        let result = Mcts::new().search_batched(
            &Nim,
            &(stones, Side::Max),
            8,
            &mut evaluator,
            simulations(3000),
        );
        assert_eq!(result.best, Some(stones % 3), "{stones} stones");
    }
}

/// Values leaves by the game's evaluation and puts most of the prior on the first move: a
/// test-only per-leaf evaluator.
struct FirstFavoured;

impl<G: Rewarded> game_tree_search::Evaluator<G> for FirstFavoured {
    fn evaluate<R: game_tree_search::Random>(
        &mut self,
        game: &G,
        position: &G::Position,
        _: &mut R,
    ) -> game_tree_search::Evaluation {
        game_tree_search::Evaluation::new(game.reward(game.evaluate(position)))
    }

    fn priors(&mut self, _: &G, _: &G::Position, moves: &[G::Move], priors: &mut Vec<f64>) {
        priors.extend((0..moves.len()).map(|i| if i == 0 { 9.0 } else { 1.0 }));
    }
}

/// An unbatched evaluator's own priors reach the batch, and steer PUCT.
#[test]
fn unbatched_evaluators_pass_their_priors_on() {
    let game = RandomTree::new(6);
    let (mut favoured, mut total) = (0, 0);
    for seed in 0..50 {
        let mut evaluator = Unbatched::new(FirstFavoured, 0);
        let mut mcts = Mcts::new().with_selection(Puct::default());
        let result = mcts.search_batched(&game, &seed, 4, &mut evaluator, simulations(200));
        if result.root.len() >= 3 {
            favoured += result.root[0].visits;
            total += result.root.iter().map(|root| root.visits).sum::<u32>();
        }
    }

    assert!(
        f64::from(favoured) > 0.4 * f64::from(total),
        "{favoured} of {total}"
    );
}

/// A batched search continues a tree kept by `advance`.
#[test]
fn batched_searches_continue_kept_trees() {
    let mut mcts = Mcts::new();
    let first = mcts.search(&Nim, &(10, Side::Max), simulations(300));
    let child = first
        .root
        .iter()
        .find(|root| root.mv == 1)
        .expect("a move")
        .visits;
    assert!(mcts.advance(&[1]));

    let mut evaluator = Unbatched::new(StaticValue, 0);
    let next = mcts.search_batched(&Nim, &(9, Side::Min), 4, &mut evaluator, simulations(40));
    assert_eq!(
        next.root.iter().map(|root| root.visits).sum::<u32>(),
        child - 1 + 40
    );
}
