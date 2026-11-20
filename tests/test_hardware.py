import unittest

from llmsim.hardware import AlphaBeta, PayloadCurve, load_gpu, load_server, parse_size

# 用户给出的 S3 实测数据（GB/s）：本卡 load / store，对端 load / store
S3_TABLE = {
    "1KiB": (0.32, 0.32, 0.34, 0.34),
    "4KiB": (1.28, 1.28, 1.32, 1.36),
    "16KiB": (4.82, 4.7, 5.4, 4.04),
    "64KiB": (18.82, 16.9, 16.4, 13.5),
    "256KiB": (39.46, 36.98, 25.6, 21.1),
    "1MiB": (46.74, 25.52, 32.67, 26.46),
    "4MiB": (38.52, 26.08, 25.55, 22.89),
}


class PayloadCurveTest(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("4KiB"), 4096)
        self.assertEqual(parse_size("1MiB"), 1024 ** 2)
        self.assertEqual(parse_size(512), 512)

    def test_interpolation_is_log_log_linear(self):
        c = load_server("s3x8").links["peer"].store
        # 32KiB 是 16KiB 与 64KiB 的几何中点，带宽为两端的几何平均
        self.assertAlmostEqual(c.bandwidth(parse_size("32KiB")) / 1e9, (4.04 * 13.5) ** 0.5, places=9)

    def test_extrapolation(self):
        c = PayloadCurve((1024.0, 4096.0), (1.0, 2.0))
        self.assertAlmostEqual(c.time(10), c.time(1024))
        self.assertAlmostEqual(c.bandwidth(1 << 30), 2.0e9)


class S3Test(unittest.TestCase):
    def test_gpu_local_curves(self):
        gpu = load_gpu("s3")
        self.assertEqual(gpu.hbm_gb, 96)
        for size, (load, store, _, _) in S3_TABLE.items():
            self.assertAlmostEqual(gpu.local.load.bandwidth(parse_size(size)) / 1e9, load, places=9)
            self.assertAlmostEqual(gpu.local.store.bandwidth(parse_size(size)) / 1e9, store, places=9)

    def test_server_peer_curves_on_every_pair(self):
        server = load_server("s3x8")
        self.assertEqual(server.n_gpus, 8)
        for size, (_, _, load, store) in S3_TABLE.items():
            b = parse_size(size)
            for a in range(8):
                for d in range(8):
                    if a != d:
                        self.assertAlmostEqual(b / server.peer_time("load", a, d, b) / 1e9, load, places=9)
                        self.assertAlmostEqual(b / server.peer_time("store", a, d, b) / 1e9, store, places=9)

    def test_no_nic(self):
        self.assertIsNone(load_server("s3x8").nic)


class GpuServerTest(unittest.TestCase):
    def test_alpha_beta_links(self):
        server = load_server("h800x8")
        self.assertIsInstance(server.gpu.local.load, AlphaBeta)
        self.assertAlmostEqual(server.peer_time("store", 0, 7, 1e6), 1e6 / 200e9 + 8e-6)
        self.assertAlmostEqual(server.local_time("load", 1e6), 1e6 / 3350e9)

    def test_every_builtin_server_loads(self):
        for name in ("s3x8", "h20x8", "h800x8", "b300x8"):
            self.assertEqual(load_server(name).n_gpus, 8)


if __name__ == "__main__":
    unittest.main()
