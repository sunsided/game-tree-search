//! Sample games for the tests: [`Nim`] with alternating sides and [`RandomTree`], whose side
//! to move is random, so one side often moves several times in a row.

use game_tree_search::{Game, Side};

/// One-pile Nim: take one or two stones; whoever cannot move has lost.
pub struct Nim;

impl Game for Nim {
    /// The stones left and the side to move.
    type Position = (u32, Side);
    /// The stones taken.
    type Move = u32;
    /// `1` if [`Side::Max`] has won, `-1` if it has lost, `0` while the game goes on.
    type Score = i32;

    fn side_to_move(&self, &(_, side): &Self::Position) -> Side {
        side
    }

    fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
        moves.extend((1..=2).filter(|&take| take <= stones));
    }

    fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
        let next = match side {
            Side::Max => Side::Min,
            Side::Min => Side::Max,
        };

        (stones - take, next)
    }

    fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
        match (stones, side) {
            (0, Side::Max) => -1,
            (0, Side::Min) => 1,
            _ => 0,
        }
    }
}

/// A game tree derived from a seed: branching, side to move and scores all come from
/// hashing the position, so the same seed always gives the same tree.
pub struct RandomTree {
    /// The most moves a position has; a position may have none and end the game.
    pub max_branching: u64,
}

/// Salt for the branching factor of a position.
const BRANCHING: u64 = 0x1;
/// Salt for the side to move in a position.
const SIDE: u64 = 0x2;
/// Salt for the score of a position.
const SCORE: u64 = 0x3;

impl Game for RandomTree {
    /// A hash of the path from the root; the root is the seed.
    type Position = u64;
    /// The index of the move among its position's moves.
    type Move = u64;
    /// A score in `-100..=100`.
    type Score = i32;

    fn side_to_move(&self, &position: &u64) -> Side {
        if mix(position ^ SIDE) % 2 == 0 {
            Side::Max
        } else {
            Side::Min
        }
    }

    fn moves(&self, &position: &u64, moves: &mut Vec<u64>) {
        moves.extend(0..mix(position ^ BRANCHING) % (self.max_branching + 1));
    }

    fn play(&self, &position: &u64, &index: &u64) -> u64 {
        mix(position.rotate_left(17) ^ index.wrapping_add(1))
    }

    fn evaluate(&self, &position: &u64) -> i32 {
        let offset = i32::try_from(mix(position ^ SCORE) % 201).expect("below 201");

        offset - 100
    }
}

/// The `SplitMix64` finalizer: a bijective mix of all 64 bits.
fn mix(value: u64) -> u64 {
    let mut z = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

    z ^ (z >> 31)
}
