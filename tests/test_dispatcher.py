import unittest

from llmsim.hardware import load_server
from llmsim.layers.dispatcher import INDEX_BYTES, TokenDispatcher, _lanes, token_bytes
from llmsim.routing import RoutingStats
from llmsim.scenario import Precision, Transfer

# 两张卡：pair[s][d] 是卡 s 发给卡 d 的 token 份数；卡 0 上 3 个本地专家，卡 1 上 2 个
PAIR = [[3, 2], [5, 1]]
EXPERT_TOKENS = [{0: 4, 1: 2, 2: 6}, {3: 2, 4: 1}]


def dispatcher(**transfer) -> TokenDispatcher:
    return TokenDispatcher(load_server("s3x8"), Transfer(**transfer), Precision(), hidden=5120, topk=6,
                           n_local_experts=3)


class TokenBytesTest(unittest.TestCase):
    def test_token_bytes(self):
        self.assertEqual(token_bytes(5120, "fp8", 128), 5120 + 40 * 4)
        self.assertEqual(token_bytes(5120, "bf16", 128), 5120 * 2)

    def test_dispatch_carries_routing_metadata(self):
        d = dispatcher()
        self.assertEqual(d.dispatch_bytes, d.hidden_bytes + 6 * (4 + 4))
        self.assertEqual(d.combine_bytes, 5120 * 2)


class AllToAllTest(unittest.TestCase):
    def setUp(self):
        self.peer = load_server("s3x8").links["peer"]

    def test_lanes(self):
        self.assertEqual(_lanes([1, 2, 3], 1), 6)
        self.assertEqual(_lanes([1, 2, 3], 3), 3)
        self.assertEqual(_lanes([4, 3, 3, 2], 2), 6)
        self.assertEqual(_lanes([], 4), 0)

    def test_push_and_pull_use_their_curves(self):
        for mode, op in (("push", "store"), ("pull", "load")):
            cost = dispatcher(mode=mode)._all_to_all([[0, 10], [10, 0]], 1024)
            self.assertAlmostEqual(max(cost.card_time), self.peer.time(op, 10 * 1024))
            self.assertAlmostEqual(cost.peer_payload_mean, 10 * 1024)

    def test_hot_receiver_is_slowest(self):
        cost = dispatcher(local_copy=False)._all_to_all([[0, 0, 50], [0, 0, 50], [1, 1, 0]], 1024)
        self.assertAlmostEqual(max(cost.card_time), 2 * self.peer.store.time(50 * 1024))

    def test_combine_is_dispatch_reversed(self):
        d, r = dispatcher(), RoutingStats("token", EXPERT_TOKENS, PAIR)
        self.assertEqual([x / d.combine_bytes for x in d.combine(r).sent],
                         [x / d.dispatch_bytes for x in d.dispatch(r).recv])


class PermuteTest(unittest.TestCase):
    def setUp(self):
        self.d = dispatcher()
        self.r = RoutingStats("token", EXPERT_TOKENS, PAIR)
        local = self.d.server.gpu.local
        self.load, self.store = local.load.time, local.store.time

    def test_permute_index_reads_routing_ids_and_writes_counts_and_index(self):
        # 卡 0 收到 3 + 5 = 8 个 token，每个带 6 个专家 id；3 个本地专家共 12 行
        expected = self.load(8 * 6 * INDEX_BYTES) + self.store(3 * INDEX_BYTES) + self.store(12 * INDEX_BYTES)
        self.assertAlmostEqual(self.d.permute(self.r).index_time[0], expected)

    def test_permute_data_reads_device_segments_and_writes_expert_segments(self):
        hb = self.d.hidden_bytes
        expected = (self.load(3 * hb) + self.load(5 * hb)
                    + self.store(4 * hb) + self.store(2 * hb) + self.store(6 * hb))
        self.assertAlmostEqual(self.d.permute(self.r).data_time[0], expected)

    def test_unpermute_reuses_index_and_moves_back(self):
        cb = self.d.combine_bytes
        cost = self.d.unpermute(self.r)
        self.assertAlmostEqual(cost.index_time[0], self.load(12 * INDEX_BYTES))
        expected = (self.load(4 * cb) + self.load(2 * cb) + self.load(6 * cb)
                    + self.store(3 * cb) + self.store(5 * cb))
        self.assertAlmostEqual(cost.data_time[0], expected)

    def test_permute_modes(self):
        hb = self.d.hidden_bytes
        buffer = dispatcher(permute="buffer").permute(self.r).data_time[0]
        self.assertAlmostEqual(buffer, self.load(8 * hb) + self.store(12 * hb))
        none = dispatcher(permute="none").permute(self.r)
        self.assertEqual(none.data_time, [0.0, 0.0])
        self.assertGreater(none.index_time[0], 0)


if __name__ == "__main__":
    unittest.main()
