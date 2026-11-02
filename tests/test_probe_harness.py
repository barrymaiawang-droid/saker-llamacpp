import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

REPRO = Path(__file__).parents[1] / "reports/item23-20261102/reproduce"
spec = importlib.util.spec_from_file_location("probe_engine", REPRO / "probe_engine.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def launch(**overrides):
    base = dict(backend="vulkan", profile="ar", ctx=16384, ubatch=256, threads=4, port=18153,
                no_retune=False, out=Path("/test/out"))
    base.update(overrides)
    return SimpleNamespace(**base)


class ServerCommandTests(unittest.TestCase):
    def test_cpu_arm_disables_every_gpu_path(self):
        cmd = probe.server_command(launch(backend="cpu", threads=16), Path("/test/llama-server"))
        self.assertEqual(cmd[cmd.index("-ngl") + 1], "0")
        for flag in ("--device", "--no-op-offload", "--no-kv-offload"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--device") + 1], "none")
        self.assertEqual(cmd[cmd.index("-t") + 1], "16")
        self.assertNotIn("--spec-type", cmd)

    def test_gpu_arms_are_unchanged_by_the_cpu_option(self):
        vulkan = probe.server_command(launch(), Path("/test/llama-server"))
        self.assertEqual(vulkan[vulkan.index("-ngl") + 1], "99")
        for flag in probe.CPU_ONLY_FLAGS:
            self.assertNotIn(flag, vulkan)
        mtp = probe.server_command(launch(profile="mtp"), Path("/test/llama-server"))
        self.assertEqual(mtp[mtp.index("--spec-draft-n-max") + 1], "2")

    def test_cpu_arm_uses_the_vulkan_build_tree(self):
        self.assertEqual(probe.server_binary(launch(backend="cpu")), probe.server_binary(launch(backend="vulkan")))
        self.assertNotEqual(probe.server_binary(launch(backend="hip")), probe.server_binary(launch(backend="cpu")))

    def test_sampling_matches_the_installed_service(self):
        cmd = probe.server_command(launch(backend="cpu"), Path("/test/llama-server"))
        for name, value in [("--temp", "1"), ("--top-k", "20"), ("--top-p", ".95"), ("--min-p", "0"), ("--repeat-penalty", "1")]:
            self.assertEqual(cmd[cmd.index(name) + 1], value)


if __name__ == "__main__":
    unittest.main()
