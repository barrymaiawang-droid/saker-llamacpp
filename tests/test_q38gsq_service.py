import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("q38gsq_service", Path(__file__).parents[1] / "tools/q38gsq_service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class LaunchTests(unittest.TestCase):
    config = {"binary": "/test/llama-server", "model": "/test/model.gguf"}

    def test_thinking_defaults_and_shortlist_removal(self):
        cmd, env = service.launch_spec(self.config, {"SAKER_SHORTLIST": "3584"})
        self.assertNotIn("SAKER_SHORTLIST", env)
        for name, value in [("--temp", "1.0"), ("--top-p", "0.95"), ("--top-k", "20"), ("--min-p", "0"), ("--repeat-penalty", "1")]:
            self.assertEqual(cmd[cmd.index(name)+1], value)
        self.assertNotIn("--spec-type", cmd)

    def test_fleet_controls_reach_process(self):
        import json
        cmd, env = service.launch_spec(self.config, {"CTX": "32768", "RBUDGET": "8192", "PROFILE": "MTP2", "REASONING": "off", "LLAMA_CHAT_TEMPLATE_KWARGS": '{"reasoning_effort":"low"}'})
        for name, value in [("-c", "32768"), ("-n", "8192"), ("--reasoning", "off"), ("--spec-type", "draft-mtp"), ("--spec-draft-n-max", "2"), ("--temp", "0.7")]:
            self.assertEqual(cmd[cmd.index(name)+1], value)
        self.assertEqual(json.loads(env["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"]), {"reasoning_effort":"low", "enable_thinking":False, "preserve_thinking":True})

    def test_explicit_greedy_sampling_is_respected(self):
        cmd, _ = service.launch_spec(self.config, {"TEMP": "0"})
        self.assertEqual(cmd[cmd.index("--temp")+1], "0")

    def test_template_kwargs_use_the_native_environment_contract(self):
        import json
        _, child = service.launch_spec(self.config, {"LLAMA_CHAT_TEMPLATE_KWARGS":'{"reasoning_effort":"medium"}'})
        self.assertEqual(json.loads(child["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"])["reasoning_effort"], "medium")

    def test_effort_config_can_be_overridden(self):
        import json
        config = dict(self.config, effort="medium")
        for env, expected in [({}, "medium"), ({"LLAMA_ARG_REASONING_EFFORT":"xhigh"}, "xhigh"),
                              ({"LLAMA_ARG_REASONING_EFFORT":"xhigh", "LLAMA_CHAT_TEMPLATE_KWARGS":'{"reasoning_effort":"low"}'}, "low"),
                              ({"LLAMA_CHAT_TEMPLATE_KWARGS":'{"reasoning_effort":"low"}'}, "low")]:
            with self.subTest(env=env):
                _, child = service.launch_spec(config, env)
                self.assertEqual(json.loads(child["LLAMA_ARG_CHAT_TEMPLATE_KWARGS"])["reasoning_effort"], expected)
                self.assertEqual(child["LLAMA_ARG_REASONING_EFFORT"], expected)

    def test_invalid_settings_fail_before_restart(self):
        for env in [{"CTX":"0"}, {"RBUDGET":"0"}, {"PROFILE":"bogus"}, {"REASONING":"bogus"}, {"LLAMA_CHAT_TEMPLATE_KWARGS":'[]'}, {"LLAMA_CHAT_TEMPLATE_KWARGS":'{"reasoning_effort":"high"}'}]:
            with self.subTest(env=env), self.assertRaises(ValueError):
                service.launch_spec(self.config, env)


if __name__ == "__main__":
    unittest.main()
