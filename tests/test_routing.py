import random
import unittest

from llmsim.routing import expert_popularity, place_experts, sample_aggregate, sample_by_token


class PlacementTest(unittest.TestCase):
    def test_eplb_slots_full_and_unique_per_rank(self):
        pop = expert_popularity(256, 1.0, random.Random(0))
        for slots in place_experts(pop, 32, 32, "eplb").slots:
            self.assertEqual(len(slots), 9)
            self.assertEqual(len(set(slots)), 9)

    def test_static_rejects_redundant(self):
        with self.assertRaises(ValueError):
            place_experts([1 / 8] * 8, 4, 4, "static")


class SamplingTest(unittest.TestCase):
    def test_token_copies_conserved_without_dedup(self):
        pop = expert_popularity(256, 0.0, random.Random(0))
        pl = place_experts(pop, 32, 0, "static")
        st = sample_by_token(pl, pop, 8, [64] * 32, "none", random.Random(1))
        self.assertEqual(sum(map(sum, st.pair)), 64 * 32 * 8)
        self.assertEqual(sum(sum(d.values()) for d in st.expert_tokens), 64 * 32 * 8)

    def test_rank_dedup_sends_fewer_copies(self):
        pop = expert_popularity(256, 0.0, random.Random(0))
        pl = place_experts(pop, 16, 0, "static")
        a = sample_by_token(pl, pop, 8, [64] * 16, "none", random.Random(2))
        b = sample_by_token(pl, pop, 8, [64] * 16, "rank", random.Random(2))
        self.assertLess(sum(map(sum, b.pair)), sum(map(sum, a.pair)))

    def test_eplb_reduces_imbalance_under_skew(self):
        pop = expert_popularity(256, 1.2, random.Random(0))
        imb = [sample_by_token(pl, pop, 8, [128] * 32, "rank", random.Random(1)).imbalance()
               for pl in (place_experts(pop, 32, 0, "static"), place_experts(pop, 32, 32, "eplb"))]
        self.assertLess(imb[1], imb[0])

    def test_aggregate_matches_token_sampling(self):
        pop = expert_popularity(256, 0.6, random.Random(0))
        pl = place_experts(pop, 16, 0, "eplb")
        for dedup in ("rank", "none"):
            a = sample_by_token(pl, pop, 8, [512] * 16, dedup, random.Random(1))
            b = sample_aggregate(pl, pop, 8, [512] * 16, dedup, random.Random(1))
            self.assertAlmostEqual(sum(map(sum, b.pair)) / sum(map(sum, a.pair)), 1.0, delta=0.05, msg=dedup)
            self.assertAlmostEqual(b.imbalance(), a.imbalance(), delta=0.1)


if __name__ == "__main__":
    unittest.main()
