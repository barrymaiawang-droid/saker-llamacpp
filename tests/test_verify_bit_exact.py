import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
import tempfile
from unittest.mock import MagicMock, patch

spec = importlib.util.spec_from_file_location("verify_bit_exact", Path(__file__).parents[1] / "tools/verify_bit_exact.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class VerdictTests(unittest.TestCase):
    def run_gate(self, results, reps=2):
        argv = ["verify_bit_exact", "--arms", "generic,saker-noshortlist", "--prompts", "think", "--reps", str(reps)]
        output = io.StringIO()
        with patch.object(verify.sys, "argv", argv), patch.object(verify, "free_port", return_value=True), \
                patch.object(verify, "run_arm", side_effect=results), contextlib.redirect_stdout(output):
            status = verify.main()
        return status, output.getvalue()

    def test_no_results_cannot_pass(self):
        status, output = self.run_gate([{}, {}])
        self.assertEqual(status, 2, output)
        self.assertNotIn("ALL ARMS BIT-IDENTICAL", output)

    def test_partial_results_cannot_pass(self):
        ref = {("think", 1): "same", ("think", 2): "same"}
        status, output = self.run_gate([ref, {("think", 1): "same"}])
        self.assertEqual(status, 2, output)

    def test_unstable_reference_is_inconclusive_even_when_reps_match(self):
        data = {("think", 1): "first", ("think", 2): "second"}
        status, output = self.run_gate([data, data])
        self.assertEqual(status, 2, output)
        self.assertNotIn("ALL ARMS BIT-IDENTICAL", output)

    def test_stable_complete_match_passes(self):
        data = {("think", 1): "same", ("think", 2): "same"}
        self.assertEqual(self.run_gate([data, data])[0], 0)

    def test_stable_complete_mismatch_fails(self):
        ref = {("think", 1): "same", ("think", 2): "same"}
        arm = {("think", 1): "wrong", ("think", 2): "wrong"}
        self.assertEqual(self.run_gate([ref, arm])[0], 1)

    def test_malformed_http_result_cannot_be_hashed_as_valid_output(self):
        proc = MagicMock()
        proc.poll.return_value = 0
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{}'
        with tempfile.TemporaryDirectory() as out, patch.object(verify, "OUT", out), \
                patch.object(verify.subprocess, "Popen", return_value=proc), \
                patch.object(verify.subprocess, "run"), patch.object(verify.time, "sleep"), \
                patch.object(verify.urllib.request, "urlopen", return_value=response), \
                contextlib.redirect_stdout(io.StringIO()):
            data = verify.run_arm("generic", "/test/binary", "", ["think"], 1, 8192, 32)
        self.assertEqual(data, {})


if __name__ == "__main__":
    unittest.main()
