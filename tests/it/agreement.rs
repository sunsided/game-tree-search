//! Cross-checks: every search agrees with plain minimax, and principal variations replay
//! to their score.

use game_tree_search::{Bound, Game, SearchResult, Side, minimax, negamax};

use crate::games::{Nim, RandomTree};

/// Seeds of the random trees the searches are compared on.
const SEEDS: core::ops::Range<u64> = 0..200;

/// The score reached by playing `result`'s principal variation from `root`.
pub fn replayed<G: Game>(
    game: &G,
    root: &G::Position,
    result: &SearchResult<G::Move, G::Score>,
) -> G::Score
where
    G::Position: Clone,
{
    let leaf = result.pv.iter().fold(root.clone(), |position, step| {
        game.play(&position, &step.mv)
    });

    game.evaluate(&leaf)
}

/// Negamax returns minimax's score and line on random trees, where one side often moves
/// several times in a row.
#[test]
fn negamax_agrees_with_minimax_on_random_trees() {
    let game = RandomTree::new(4);
    for seed in SEEDS {
        for depth in 0..=5 {
            let reference = minimax(&game, &seed, depth);
            let result = negamax(&game, &seed, depth);
            assert_eq!(result, reference, "seed {seed}, depth {depth}");
        }
    }
}

/// The random trees do contain consecutive moves by the same side, so the agreement above
/// covers non-alternating play.
#[test]
fn random_trees_repeat_the_side_to_move() {
    let game = RandomTree::new(4);
    let repeats = SEEDS.filter(|&seed| {
        let mut moves = Vec::new();
        game.moves(&seed, &mut moves);
        moves.iter().any(|mv| {
            let child = game.play(&seed, mv);
            let mut grandchildren = Vec::new();
            game.moves(&child, &mut grandchildren);
            !grandchildren.is_empty() && game.side_to_move(&child) == game.side_to_move(&seed)
        })
    });
    assert!(repeats.count() > 0);
}

/// The principal variation of every search replays to the score it reports.
#[test]
fn principal_variations_replay_to_their_score() {
    let game = RandomTree::new(4);
    for seed in SEEDS {
        for result in [minimax(&game, &seed, 5), negamax(&game, &seed, 5)] {
            assert_eq!(replayed(&game, &seed, &result), result.score, "seed {seed}");
        }
    }
}

/// Every step of a reference principal variation carries the exact minimax value of the
/// position it leads to.
#[test]
fn reference_lines_score_every_step_exactly() {
    let game = RandomTree::new(4);
    for seed in SEEDS {
        let result = minimax(&game, &seed, 5);
        let mut position = seed;
        for (ply, step) in (1..).zip(&result.pv) {
            position = game.play(&position, &step.mv);
            let value = minimax(&game, &position, 5 - ply).score;
            assert_eq!(
                (step.score, step.bound),
                (value, Bound::Exact),
                "seed {seed}, ply {ply}"
            );
        }
    }
}

/// Searched to the end, Nim is lost for the side to move exactly on multiples of three.
#[test]
fn nim_is_lost_on_multiples_of_three() {
    for stones in 0..=12 {
        for side in [Side::Max, Side::Min] {
            let result = negamax(&Nim, &(stones, side), 12);
            let mover_wins = stones % 3 != 0;
            let expected = if mover_wins == (side == Side::Max) {
                1
            } else {
                -1
            };
            assert_eq!(result.score, expected, "{stones} stones, {side:?} to move");
            assert_eq!(result, minimax(&Nim, &(stones, side), 12));
        }
    }
}

/// Depth `0` evaluates the root without searching, and a finished game has no best move.
#[test]
fn depth_zero_and_finished_games_have_no_line() {
    let shallow = minimax(&Nim, &(5, Side::Max), 0);
    assert_eq!(
        (shallow.score, shallow.best(), shallow.stats.nodes),
        (0, None, 1)
    );

    let finished = negamax(&Nim, &(0, Side::Min), 4);
    assert_eq!((finished.score, finished.best()), (1, None));
}
