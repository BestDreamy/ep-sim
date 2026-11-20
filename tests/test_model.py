import functools
import unittest
from pathlib import Path

from llmsim.config import load_config
from llmsim.model import Transformer
from llmsim.runner import simulate
from llmsim.scenario import apply_override, load_scenario_dict, scenario_from_dict

SCENARIO = Path(__file__).resolve().parent.parent / "configs" / "scenarios" / "deepseek_v41_s3.toml"


def scenario(**overrides):
    d = load_scenario_dict(SCENARIO)
    for k, v in overrides.items():
        d = apply_override(d, k.replace("__", "."), v)
    return scenario_from_dict(d)


def run(**overrides):
    return _run(tuple(sorted(overrides.items())))


@functools.cache
def _run(overrides: tuple):
    return simulate(scenario(**dict(overrides)))


class ConfigTest(unittest.TestCase):
    def test_v41_moe_fields(self):
        m = load_config("deepseek_v4_1")
        self.assertEqual((m.hidden, m.n_experts, m.topk, m.intermediate, m.n_shared), (5120, 384, 6, 2304, 1))
        self.assertEqual(m.n_moe_layers, 40)
        self.assertEqual(m.expert_dtype, "fp4")


class TransformerTest(unittest.TestCase):
    def test_one_moe_layer_per_model_layer(self):
        model = Transformer(scenario())
        self.assertEqual([l.layer_id for l in model.layers], list(range(40)))

    def test_param_bytes(self):
        params = run().param_bytes
        self.assertAlmostEqual(params["routed_experts"], 48 * 3 * 5120 * 2304 * 40 * (0.5 + 1 / 32))
        self.assertAlmostEqual(params["shared_experts"], 3 * 5120 * 2304 * 40 * 1.0)
        self.assertGreater(run(parallel__ep=4).param_bytes["routed_experts"], params["routed_experts"])

    def test_ep_beyond_server_is_rejected(self):
        with self.assertRaises(ValueError):
            scenario(parallel__ep=16)


class ForwardTest(unittest.TestCase):
    def test_default_scenario(self):
        prefill, decode = run().phases
        self.assertEqual([ph.layers[0].routing.method for ph in (prefill, decode)], ["aggregate", "token"])
        self.assertGreater(prefill.layers[0].dispatch.peer_payload_mean, decode.layers[0].dispatch.peer_payload_mean)
        self.assertGreater(prefill.layer_time, decode.layer_time)

    def test_step_time_is_sum_of_layers(self):
        for ph in run().phases:
            self.assertEqual(len(ph.layers), 40)
            self.assertAlmostEqual(ph.time, sum(l.time for l in ph.layers))

    def test_layer_time_is_slowest_card_per_stage(self):
        l = run().phases[1].layers[0]
        dispatch = max(a + b for a, b in zip(l.dispatch.card_time, l.permute.card_time))
        combine = max(a + b for a, b in zip(l.unpermute.card_time, l.combine.card_time))
        self.assertAlmostEqual(l.time, dispatch + combine)

    def test_layer_bytes_match_routing(self):
        r = run()
        l = r.phases[1].layers[0]
        bpt = r.token_bytes["dispatch"]
        for s, row in enumerate(l.routing.pair):
            self.assertAlmostEqual(l.dispatch.sent[s], (sum(row) - row[s]) * bpt)
            self.assertAlmostEqual(l.dispatch.local[s], row[s] * bpt)

    def test_permute_none_keeps_index_cost(self):
        with_data, without = run().phases[1], run(a2a__permute="none").phases[1]
        self.assertLess(without.time, with_data.time)
        self.assertEqual(without.layers[0].permute.data_time, [0.0] * 8)
        self.assertGreater(min(without.layers[0].permute.index_time), 0)

    def test_decode_payload_shrinks_as_ep_grows(self):
        p = [run(parallel__ep=ep).phases[1].layers[0].dispatch.peer_payload_mean for ep in (2, 4, 8)]
        self.assertGreater(p[0], p[1])
        self.assertGreater(p[1], p[2])

    def test_more_concurrency_is_faster(self):
        self.assertLess(run(a2a__concurrency=8).phases[1].time, run().phases[1].time)

    def test_gpu_server(self):
        self.assertGreater(run(server="h800x8").phases[0].time, 0)


if __name__ == "__main__":
    unittest.main()
