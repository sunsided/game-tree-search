//! [`SharedTable`]: one transposition table for several threads (feature `parallel`).

use alloc::vec::Vec;
use std::sync::{Mutex, MutexGuard, PoisonError};

use super::{Entry, SlotTable, Table, TableKey};

/// A transposition table several threads search with at once: [`SlotTable`] shards, each
/// behind its own lock.
///
/// A key's shard comes from the high bits of [`TableKey::slot`], its slot within the shard
/// from the low bits, so threads rarely wait for the same lock. Searchers use it through a
/// shared reference: `&SharedTable` implements [`Table`], so each thread builds its own
/// [`Searcher::with_table`](crate::Searcher::with_table)`(&table)`, as
/// [`deepen_shared`](crate::deepen_shared()) does for Lazy SMP.
///
/// Available with the `parallel` feature, which needs `std`.
#[derive(Debug)]
pub struct SharedTable<K, M, S> {
    /// The shards; their number is a power of two.
    shards: Vec<Mutex<SlotTable<K, M, S>>>,
}

impl<K: TableKey, M: Clone, S: Clone> SharedTable<K, M, S> {
    /// A table with about `capacity` slots in `shards` shards, both rounded up to powers of
    /// two (at least one).
    ///
    /// # Panics
    ///
    /// Panics if a rounded size does not fit in `usize`.
    #[must_use]
    pub fn new(capacity: usize, shards: usize) -> Self {
        let shards = shards
            .max(1)
            .checked_next_power_of_two()
            .expect("the shard count fits in usize when rounded up");
        let per_shard = capacity.div_ceil(shards);

        Self {
            shards: (0..shards)
                .map(|_| Mutex::new(SlotTable::new(per_shard)))
                .collect(),
        }
    }

    /// The number of slots over all shards.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.shards.len() * lock(&self.shards[0]).capacity()
    }

    /// Empties every shard, for example before searching a different game.
    pub fn clear(&self) {
        for shard in &self.shards {
            lock(shard).clear();
        }
    }

    /// The shard of `key`: its high bits, as many as the shard count needs.
    fn shard(&self, key: &K) -> MutexGuard<'_, SlotTable<K, M, S>> {
        let mask = u64::try_from(self.shards.len() - 1).expect("a shard count fits in u64");
        let index = usize::try_from(key.slot().rotate_right(32) & mask)
            .expect("a masked shard index is below the shard count");

        lock(&self.shards[index])
    }
}

/// Locks `shard`, ignoring a poisoned lock: a thread that panicked mid-store left at worst one
/// stale slot, which the key check still guards.
fn lock<T>(shard: &Mutex<T>) -> MutexGuard<'_, T> {
    shard.lock().unwrap_or_else(PoisonError::into_inner)
}

impl<K: TableKey, M: Clone, S: Clone> Table<K, M, S> for &SharedTable<K, M, S> {
    fn probe(&mut self, key: &K) -> Option<Entry<M, S>> {
        self.shard(key).probe(key)
    }

    fn store(&mut self, key: K, entry: Entry<M, S>) {
        self.shard(&key).store(key, entry);
    }
}
