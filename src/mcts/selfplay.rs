//! Self-play support: Dirichlet noise on the root's priors, the visit distribution as a
//! training target, and move choice by temperature.

use alloc::vec::Vec;
use core::f64::consts::TAU;

use super::MctsResult;
use super::random::Random;

/// Dirichlet noise mixed into the root's move priors, as `AlphaZero` explores in self-play.
///
/// Each search draws `η ~ Dirichlet(alpha)` over the root's moves and sets every root
/// prior to `(1 - fraction) * prior + fraction * η`, so even moves the priors rule out get
/// tried. Only rules that read priors, such as [`Puct`](crate::Puct), are affected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootNoise {
    /// The concentration `alpha`, positive: small values put the noise on a few moves.
    /// `AlphaZero` used about `10 / moves`: `0.3` for chess, `0.03` for Go.
    pub alpha: f64,
    /// The share of the noise in each prior, in `[0, 1]`; `0.25` as in `AlphaZero`.
    pub fraction: f64,
}

impl Default for RootNoise {
    /// `alpha = 0.3`, `fraction = 0.25`, `AlphaZero`'s settings for chess.
    fn default() -> Self {
        Self {
            alpha: 0.3,
            fraction: 0.25,
        }
    }
}

impl RootNoise {
    /// Mixes fresh noise drawn from `rng` into `priors`, which sum to one.
    pub(super) fn mix<R: Random>(&self, priors: &mut [f64], rng: &mut R) {
        let mut noise = Vec::with_capacity(priors.len());
        dirichlet(self.alpha, priors.len(), rng, &mut noise);
        for (prior, eta) in priors.iter_mut().zip(noise) {
            *prior = (1.0 - self.fraction) * *prior + self.fraction * eta;
        }
    }
}

impl<M> MctsResult<M> {
    /// The search's move distribution π: each root move's share of the root visits, in root
    /// move order. Uniform before the first simulation; empty without moves.
    ///
    /// With [`MctsResult::value`] and the game's final outcome, the training target of
    /// `AlphaZero`-style self-play.
    #[must_use]
    pub fn policy(&self) -> Vec<f64> {
        let total: f64 = self.root.iter().map(|root| f64::from(root.visits)).sum();
        if total == 0.0 {
            let uniform =
                1.0 / f64::from(u32::try_from(self.root.len().max(1)).unwrap_or(u32::MAX));
            return self.root.iter().map(|_| uniform).collect();
        }

        self.root
            .iter()
            .map(|root| f64::from(root.visits) / total)
            .collect()
    }

    /// A root move drawn with probability proportional to `visits ^ (1 / temperature)`.
    ///
    /// A `temperature` of `1` plays in proportion to the visits, as `AlphaZero` does in the
    /// first moves of self-play; lower values favour the most visited moves; `0` or less
    /// plays [`MctsResult::best`]. `None` if the root has no moves.
    ///
    /// # Examples
    ///
    /// ```
    /// # use game_tree_search::{Game, Side};
    /// # struct Nim;
    /// # impl Game for Nim {
    /// #     type Position = (u32, Side);
    /// #     type Move = u32;
    /// #     type Score = i32;
    /// #     fn side_to_move(&self, &(_, side): &Self::Position) -> Side { side }
    /// #     fn moves(&self, &(stones, _): &Self::Position, moves: &mut Vec<u32>) {
    /// #         moves.extend((1..=2).filter(|&take| take <= stones));
    /// #     }
    /// #     fn play(&self, &(stones, side): &Self::Position, take: &u32) -> Self::Position {
    /// #         (stones - take, if side == Side::Max { Side::Min } else { Side::Max })
    /// #     }
    /// #     fn evaluate(&self, &(stones, side): &Self::Position) -> i32 {
    /// #         match (stones, side) { (0, Side::Max) => -1, (0, Side::Min) => 1, _ => 0 }
    /// #     }
    /// # }
    /// # impl game_tree_search::Rewarded for Nim {
    /// #     fn reward(&self, score: i32) -> f64 { f64::from(score + 1) / 2.0 }
    /// # }
    /// use game_tree_search::{Mcts, SplitMix64, TreeStats};
    ///
    /// let budget = |stats: &TreeStats| stats.simulations >= 2000;
    /// let result = Mcts::new().search(&Nim, &(7, Side::Max), budget);
    /// let policy = result.policy();
    /// assert!((policy.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    /// assert!(result.value > 0.5);
    ///
    /// let mut rng = SplitMix64::new(1);
    /// assert_eq!(result.sample(0.0, &mut rng), result.best.as_ref());
    /// assert!(result.sample(1.0, &mut rng).is_some());
    /// ```
    pub fn sample<R: Random>(&self, temperature: f64, rng: &mut R) -> Option<&M> {
        if temperature <= 0.0 || self.root.is_empty() {
            return self
                .best
                .as_ref()
                .or(self.root.first().map(|root| &root.mv));
        }

        let most = self
            .root
            .iter()
            .map(|root| root.visits)
            .max()
            .unwrap_or(0)
            .max(1);
        let weight = |visits: u32| {
            if visits == 0 {
                return 0.0;
            }

            libm::exp((libm::log(f64::from(visits)) - libm::log(f64::from(most))) / temperature)
        };
        let total: f64 = self.root.iter().map(|root| weight(root.visits)).sum();
        if total == 0.0 {
            return self
                .root
                .get(rng.below(self.root.len()))
                .map(|root| &root.mv);
        }

        // Rounding can leave a sliver of the draw after the last weight: it goes to the last
        // visited move.
        let last = self.root.iter().rposition(|root| root.visits > 0)?;
        let mut draw = unit(rng) * total;
        let index = self
            .root
            .iter()
            .position(|root| {
                draw -= weight(root.visits);
                draw < 0.0
            })
            .unwrap_or(last);

        Some(&self.root[index].mv)
    }
}

/// A uniform draw from `[0, 1)` with 53 random bits, divided by `2^53`.
fn unit<R: Random>(rng: &mut R) -> f64 {
    #[expect(clippy::cast_precision_loss, reason = "53 bits fit an f64 exactly")]
    let bits = (rng.next_u64() >> 11) as f64;

    bits / 9_007_199_254_740_992.0
}

/// Appends a draw from the symmetric Dirichlet distribution with concentration `alpha` over
/// `n` outcomes to `out`: normalized gamma draws.
fn dirichlet<R: Random>(alpha: f64, n: usize, rng: &mut R, out: &mut Vec<f64>) {
    let start = out.len();
    out.extend((0..n).map(|_| gamma(alpha, rng)));
    let total: f64 = out[start..].iter().sum();
    if total > 0.0 {
        for draw in &mut out[start..] {
            *draw /= total;
        }
    }
}

/// A draw from the gamma distribution with shape `shape` and scale 1 (Marsaglia and Tsang
/// 2000), boosted for shapes below 1.
fn gamma<R: Random>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 {
        let boost = libm::pow(1.0 - unit(rng), 1.0 / shape);
        return gamma(shape + 1.0, rng) * boost;
    }

    let base = shape - 1.0 / 3.0;
    let scale = 1.0 / libm::sqrt(9.0 * base);
    loop {
        let normal = normal(rng);
        let cube = 1.0 + scale * normal;
        if cube <= 0.0 {
            continue;
        }

        let cube = cube * cube * cube;
        let uniform = 1.0 - unit(rng);
        let bound = 0.5 * normal * normal + base - base * cube + base * libm::log(cube);
        if libm::log(uniform) < bound {
            return base * cube;
        }
    }
}

/// A standard normal draw (Box and Muller 1958).
fn normal<R: Random>(rng: &mut R) -> f64 {
    let (u, v) = (1.0 - unit(rng), unit(rng));

    libm::sqrt(-2.0 * libm::log(u)) * libm::cos(TAU * v)
}

/// Unit tests for the noise and the temperature sampling.
#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use super::{RootNoise, dirichlet, gamma, unit};
    use crate::SplitMix64;
    use crate::mcts::{MctsResult, MoveStats, TreeStats};

    /// A result with root moves `0..` visited `visits` times.
    fn result(visits: &[u32]) -> MctsResult<usize> {
        let root: Vec<_> = (0..)
            .zip(visits)
            .map(|(mv, &visits)| MoveStats {
                mv,
                visits,
                value: 0.0,
            })
            .collect();
        let best =
            (0..visits.len()).reduce(|best, i| if visits[i] > visits[best] { i } else { best });

        MctsResult {
            best,
            root,
            pv: Vec::new(),
            value: 0.5,
            stats: TreeStats::default(),
        }
    }

    /// Dirichlet draws are probabilities; their mean is uniform, and a small concentration
    /// piles them onto one outcome.
    #[test]
    fn dirichlet_draws_are_probabilities() {
        let mut rng = SplitMix64::new(3);
        let (mut means, mut peaks) = (vec![0.0; 4], [0.0; 2]);
        for _ in 0..2000 {
            for (peak, alpha) in peaks.iter_mut().zip([0.05, 5.0]) {
                let mut draw = Vec::new();
                dirichlet(alpha, 4, &mut rng, &mut draw);
                assert!((draw.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(draw.iter().all(|&p| (0.0..=1.0).contains(&p)));
                *peak += draw.iter().copied().fold(0.0, f64::max) / 2000.0;
                if alpha > 1.0 {
                    for (mean, p) in means.iter_mut().zip(&draw) {
                        *mean += p / 2000.0;
                    }
                }
            }
        }

        assert!(
            means.iter().all(|&mean| (mean - 0.25).abs() < 0.02),
            "{means:?}"
        );
        assert!(peaks[0] > 0.8 && peaks[1] < 0.5, "{peaks:?}");
    }

    /// Gamma draws have the shape as their mean, below and above one.
    #[test]
    fn gamma_draws_have_the_shape_as_mean() {
        let mut rng = SplitMix64::new(5);
        for shape in [0.3, 2.5] {
            let mean = (0..20_000).map(|_| gamma(shape, &mut rng)).sum::<f64>() / 20_000.0;
            assert!(
                (mean - shape).abs() < 0.05 * shape.max(1.0),
                "{shape}: {mean}"
            );
        }

        assert!((0..1000).all(|_| (0.0..1.0).contains(&unit(&mut rng))));
    }

    /// Mixing keeps priors summing to one and moves them towards the noise.
    #[test]
    fn noise_keeps_priors_probabilities() {
        let mut priors = vec![1.0, 0.0, 0.0];
        RootNoise {
            alpha: 0.3,
            fraction: 0.5,
        }
        .mix(&mut priors, &mut SplitMix64::new(1));
        assert!((priors.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(priors[0] >= 0.5 && priors[0] < 1.0);
        assert_eq!(
            RootNoise::default(),
            RootNoise {
                alpha: 0.3,
                fraction: 0.25
            }
        );
    }

    /// The policy follows the visits; sampling at temperature 0 plays the best move, at 1
    /// in proportion to the visits, and never an unvisited move.
    #[test]
    fn policy_and_temperature_follow_the_visits() {
        let visited = result(&[30, 60, 10, 0]);
        assert_eq!(visited.policy(), [0.3, 0.6, 0.1, 0.0]);
        assert_eq!(result(&[0, 0]).policy(), [0.5, 0.5]);
        assert_eq!(result(&[]).policy(), Vec::<f64>::new());

        let mut rng = SplitMix64::new(9);
        assert_eq!(visited.sample(0.0, &mut rng), Some(&1));
        let mut counts = [0_u32; 4];
        for _ in 0..10_000 {
            counts[*visited.sample(1.0, &mut rng).expect("a move")] += 1;
        }

        assert_eq!(counts[3], 0);
        assert!(
            (f64::from(counts[1]) / 10_000.0 - 0.6).abs() < 0.03,
            "{counts:?}"
        );
        let sharp = (0..1000)
            .filter(|_| visited.sample(0.1, &mut rng) == Some(&1))
            .count();
        assert!(sharp > 990, "{sharp}");

        assert_eq!(result(&[]).sample(1.0, &mut rng), None);
        assert!(result(&[0, 0]).sample(1.0, &mut rng).is_some());
    }
}
