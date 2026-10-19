"""Tests for the Saker guard.

These run against a fake upstream, so they prove the guard's own logic: detection, the recovery
ladder, the request shapes it emits, and the ways it refuses to leak or silently degrade.

They do NOT prove the interaction with a real engine. Two behaviours were only discovered by running
against llama.cpp and are modelled here deliberately, so the tests stay honest:

  * the engine echoes the WHOLE assistant message on `continue_final_message`, not just the new
    tokens (measured: sent 21 chars, got 57 back, `completion_tokens=14`) - so `fake_upstream`
    always echoes, and the guard must collapse it
  * the model can spend its entire token budget thinking, returning empty content with
    `finish_reason=length`

A test that passes here but has never been run against the engine proves less than it looks like.
"""

import contextlib
import json
import select
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from guard.saker_guard import Config, compact_exact, detect_loop, make_server, parse_cli


@contextlib.contextmanager
def running(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def fake_upstream(scenarios):
    """Minimal SSE llama-server stand-in. `scenarios[i]` is served on request i.

    Per scenario: `deltas` are the chunks to stream, `finish` the terminal reason (default "stop"),
    `omit_done` to truncate without `[DONE]`, `started`/`cancelled` events to observe cancellation.
    Echoing on `continue_final_message` is unconditional because the real engine always does it.
    """
    requests = []

    class FakeHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            payload = json.dumps({"object": "list",
                                  "data": [{"id": "saker-swift", "object": "model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            scenario = scenarios[min(len(requests) - 1, len(scenarios) - 1)]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            try:
                # Measured llama.cpp behaviour: continue_final_message returns the WHOLE assistant
                # message, not only the newly generated tokens. The fake always mirrors that.
                channel = body.get("continue_final_message")
                if isinstance(channel, str):
                    for message in reversed(body.get("messages", [])):
                        if message.get("role") == "assistant":
                            if message.get(channel):
                                frame = {"id": "chatcmpl-test", "created": 123, "model": "swift",
                                         "choices": [{"index": 0,
                                                      "delta": {channel: message[channel]},
                                                      "finish_reason": None}]}
                                self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                                self.wfile.flush()
                            break
                if scenario.get("hold"):
                    scenario["hold"].wait(20)  # keep the request in flight so concurrency is observable
                for delta in scenario["deltas"]:
                    frame = {"id": "chatcmpl-test", "created": 123, "model": "swift",
                             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                    self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                    self.wfile.flush()
                if scenario.get("started"):
                    scenario["started"].set()
                if "cancelled" in scenario:
                    ready, _, _ = select.select([self.connection], [], [], 3)
                    if ready and self.connection.recv(1, socket.MSG_PEEK) == b"":
                        scenario["cancelled"].set()
                    return
                if scenario.get("omit_done"):
                    return
                frame = {"choices": [{"index": 0, "delta": {},
                                      "finish_reason": scenario.get("finish", "stop")}],
                         "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}}
                self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\ndata: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    return ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler), requests


def chat(url, **overrides):
    body = {"messages": [{"role": "user", "content": "Explain the result."}], "max_tokens": 256}
    body.update(overrides)
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read(), error.headers


class ConfigurationTests(unittest.TestCase):
    """CLI parsing and validated settings, including that the show/hide switch is independent of
    whether reasoning actually runs."""

    def test_show_hide_switch_is_independent_of_reasoning(self):
        _, hidden = parse_cli(["--hide-thinking"])
        _, shown = parse_cli(["--show-thinking"])
        self.assertTrue(hidden.hide_thinking)
        self.assertFalse(shown.hide_thinking)
        address, _ = parse_cli(["--host", "127.0.0.1", "--port", "18814"])
        self.assertEqual(address, ("127.0.0.1", 18814))


class DetectionTests(unittest.TestCase):
    """The detector on its own: what counts as a loop, and what must never count as one."""

    def test_exact_loop_compacts_without_losing_clean_prefix(self):
        prefix = "The result is 281. Next I will explain why.\n"
        unit = "Hmm. Let me consider that again. "
        text = prefix + unit * 12
        loop = detect_loop(text)
        self.assertIsNotNone(loop)
        self.assertEqual(loop.kind, "exact")
        self.assertEqual(compact_exact(text, loop), prefix + unit)


    def test_legitimate_formatting_and_numbered_progress_do_not_trigger(self):
        self.assertIsNone(detect_loop("-" * 2048))
        self.assertIsNone(detect_loop((" " + chr(10)) * 1000))
        text = "\n".join(f"The measurement for sample {i} contains a distinct verified result that we must preserve in full."
                         for i in range(1, 10))
        self.assertIsNone(detect_loop(text))


class ProxyTests(unittest.TestCase):
    """End-to-end behaviour through the real HTTP handler: recovery, refusal, cancellation, and the
    no-leak guarantees."""

    def test_shorter_repeat_run_is_caught_once_a_recovery_has_already_happened(self):
        # Probe B2 live: after one compaction the model emitted only 3 copies and stopped, so the
        # 4-repeat threshold never fired and the caller got a useless 200. After a recovery the
        # threshold drops to 3, which escalates to the restart instead.
        unit = "I am still working through this step\n"
        upstream, requests = fake_upstream([
            {"deltas": [{"content": unit * 12}]},
            {"deltas": [{"content": unit * 3}]},
            {"deltas": [{"content": "The answer is 150 minutes."}]},
        ])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 200, raw)
        body = json.loads(raw)
        guard = body["saker_guard"]
        self.assertEqual(guard["recoveries"], 2)
        self.assertNotIn("incomplete", guard)
        content = body["choices"][0]["message"]["content"]
        self.assertLessEqual(content.count(unit.strip()), 3)
        self.assertIn("150", content)
        # recovery 1 = continue, recovery 2 = restart
        self.assertEqual(requests[1]["continue_final_message"], "content")
        self.assertNotIn("continue_final_message", requests[2])

    def test_escalation_then_best_effort_partial_instead_of_discarding_everything(self):
        unit = "Let me reconsider this again. "
        upstream, requests = fake_upstream([{"deltas": [{"reasoning_content": unit * 12}]}])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0),
                                     Config(upstream=upstream_url, hide_thinking=False))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 200, raw)
        guard = json.loads(raw)["saker_guard"]
        self.assertEqual(guard["recoveries"], 2)
        self.assertEqual(guard["incomplete"], "loop_recovery_exhausted")
        # The repetition is stripped even on the partial path, so it never reaches the caller in bulk.
        self.assertLessEqual(raw.count(b"Let me reconsider"), 2)
        self.assertEqual(len(requests), 3)
        # Ladder: recovery 1 continues the message; recovery 2 restarts without anchoring to it.
        self.assertEqual(requests[1]["continue_final_message"], "reasoning_content")
        self.assertFalse(requests[1]["add_generation_prompt"])
        self.assertNotIn("continue_final_message", requests[2])
        self.assertTrue(requests[2]["add_generation_prompt"])
        # The restart carries preserved progress as context, not as a message to continue.
        self.assertIn(unit.strip(), requests[2]["messages"][-1]["content"])
        # Both retries carry the operator's manual reminder: stop the turn, say it is looping.
        # (the continuation appends the assistant message after it; the restart does not)
        for attempt in requests[1:]:
            tail = " ".join(m.get("content") or "" for m in attempt["messages"][-2:])
            self.assertIn(Config().nudge, tail)

    def test_nudge_is_configurable_and_sent_verbatim(self):
        _, config = parse_cli(["--nudge", "you are looped"])
        self.assertEqual(config.nudge, "you are looped")
        upstream, requests = fake_upstream([
            {"deltas": [{"content": "same unit of text here. " * 12}]},
            {"deltas": [{"content": "Recovered."}]},
        ])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0),
                                     Config(upstream=upstream_url, nudge="you are looped"))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 200, raw)
        self.assertEqual(requests[1]["messages"][-2]["content"], "you are looped")

    def test_models_endpoint_is_proxied_so_a_client_can_discover_the_server(self):
        # Regression: the guard originally 501'd every GET, so a client that lists models before
        # chatting saw a broken endpoint.
        upstream, _ = fake_upstream([{"deltas": [{"content": "x"}]}])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
                    status, body = response.status, json.loads(response.read())
        self.assertEqual(status, 200)
        self.assertEqual(body["data"][0]["id"], "saker-swift")

    def test_unguarded_generation_route_is_still_refused(self):
        upstream, _ = fake_upstream([{"deltas": [{"content": "x"}]}])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                self.assertEqual(chat(url)[0], 200)  # the guarded route still works
                request = urllib.request.Request(url + "/v1/completions", b'{"prompt":"x"}',
                                                 {"Content-Type": "application/json"})
                try:
                    with urllib.request.urlopen(request, timeout=10) as response:
                        code, raw = response.status, response.read()
                except urllib.error.HTTPError as error:
                    with error:
                        code, raw = error.code, error.read()
        self.assertEqual(code, 404)
        self.assertEqual(json.loads(raw)["error"]["code"], "not_proxied")

    def test_concurrent_requests_are_served_up_to_the_limit_then_refused(self):
        # Live: a single permit made the GUARD the bottleneck. dsh held one generation open, so every
        # other request got 503 while llama-server still had n_slots = 4 free.
        gate = threading.Event()
        upstream, _ = fake_upstream([{"deltas": [{"content": "ok"}], "hold": gate}])
        with running(upstream) as upstream_url:
            server = make_server(("127.0.0.1", 0), Config(upstream=upstream_url, max_concurrent=2))
            with running(server) as url:
                results, lock, start = [], threading.Lock(), threading.Barrier(3)

                def fire():
                    start.wait(5)
                    code = chat(url)[0]
                    with lock:
                        results.append(code)

                threads = [threading.Thread(target=fire) for _ in range(3)]
                for thread in threads:
                    thread.start()
                time.sleep(0.8)  # two are held in flight; the third cannot acquire
                with lock:
                    refused = results.count(503)
                gate.set()
                for thread in threads:
                    thread.join(timeout=15)
        self.assertEqual(refused, 1, "expected exactly one 503 at max_concurrent=2, got %r" % results)
        self.assertEqual(sorted(results), [200, 200, 503])

    def test_structured_output_is_never_compacted(self):
        for extra, prefix in [({"response_format": {"type": "json_object"}}, ""), ({}, "```python\n")]:
            with self.subTest(extra=extra, prefix=prefix):
                upstream, requests = fake_upstream([{"deltas": [{"content": prefix + "same code statement; " * 20}]}])
                with running(upstream) as upstream_url:
                    with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                        status, raw, _ = chat(url, **extra)
                self.assertEqual(status, 502)
                self.assertEqual(json.loads(raw)["error"]["code"], "unsafe_compaction")
                self.assertEqual(len(requests), 1)

    def test_loop_cancellation_closes_upstream_before_continuing(self):
        cancelled = threading.Event()
        upstream, requests = fake_upstream([
            {"deltas": [{"reasoning_content": "Let me reconsider this again. " * 12}], "cancelled": cancelled},
            {"deltas": [{"content": "Recovered answer."}]},
        ])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
                self.assertTrue(cancelled.wait(1), "abandoned upstream socket was not closed")
        self.assertEqual(status, 200, raw)
        self.assertEqual(len(requests), 2)

    def test_unparsed_thinking_is_not_leaked_when_hidden(self):
        upstream, _ = fake_upstream([{"deltas": [{"content": "<think>PRIVATE</think>The answer."}]}])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(raw)["error"]["code"], "unparsed_reasoning")
        self.assertNotIn(b"PRIVATE", raw)

    def test_deadline_cancels_an_idle_upstream(self):
        cancelled = threading.Event()
        upstream, _ = fake_upstream([{"deltas": [{"role": "assistant"}], "cancelled": cancelled}])
        with running(upstream) as upstream_url:
            config = Config(upstream=upstream_url, timeout=0.25, idle_timeout=10)
            with running(make_server(("127.0.0.1", 0), config)) as url:
                start = time.monotonic()
                status, raw, _ = chat(url)
                elapsed = time.monotonic() - start
                self.assertLess(elapsed, 1.5)
                self.assertTrue(cancelled.wait(1))
        self.assertEqual(status, 504, raw)
        self.assertEqual(json.loads(raw)["error"]["code"], "deadline_exceeded")

    def test_client_disconnect_cancels_idle_generation(self):
        started, cancelled = threading.Event(), threading.Event()
        upstream, _ = fake_upstream([{"deltas": [{"role": "assistant"}], "started": started, "cancelled": cancelled}])
        with running(upstream) as upstream_url:
            server = make_server(("127.0.0.1", 0), Config(upstream=upstream_url))
            with running(server):
                client = socket.create_connection(("127.0.0.1", server.server_port))
                payload = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
                client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                               + str(len(payload)).encode() + b"\r\n\r\n" + payload)
                self.assertTrue(started.wait(1))
                client.close()
                self.assertTrue(cancelled.wait(1), "generation continued after the caller disconnected")

    def test_incomplete_stream_is_not_reported_as_success(self):
        upstream, _ = fake_upstream([{"deltas": [{"content": "Partial answer"}], "omit_done": True}])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(raw)["error"]["code"], "incomplete_stream")

    def test_exact_loop_is_compacted_and_continued_without_a_summary_call(self):
        prefix = "We have established that the answer is 281.\n"
        unit = "Let me consider the next step again. "
        suffix = "Here is the proof."
        upstream, requests = fake_upstream([
            {"deltas": [{"content": prefix}, {"content": unit * 12}]},
            {"deltas": [{"content": suffix}]},
        ])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 200, raw)
        result = json.loads(raw)
        self.assertEqual(result["choices"][0]["message"]["content"], prefix + unit + suffix)
        self.assertEqual(result["saker_guard"]["recoveries"], 1)
        self.assertEqual(len(requests), 2)
        continuation = requests[1]
        self.assertEqual(continuation["messages"][0], requests[0]["messages"][0])
        self.assertEqual(continuation["messages"][-1]["content"], prefix + unit)
        self.assertEqual(continuation["continue_final_message"], "content")
        self.assertFalse(continuation["add_generation_prompt"])
        self.assertFalse(continuation["cache_prompt"])
        self.assertEqual(continuation["messages"][-2]["content"], Config().nudge)

    def test_ramblings_use_one_bounded_summary_then_continue_from_cleaned_reasoning(self):
        prefix = "The computed count is 281.\n"
        tail = "\n".join(
            f"Let me reconsider the same argument {word} before deciding whether we should proceed with the calculation once more."
            for word in ["carefully", "closely", "deeply", "thoroughly", "patiently", "slowly"]
        )
        summary = "The count is established; the remaining task is to explain the proof."
        upstream, requests = fake_upstream([
            {"deltas": [{"reasoning_content": prefix + tail}]},
            {"deltas": [{"content": summary}]},
            {"deltas": [{"reasoning_content": "Now use the recurrence."}, {"content": "The answer is 281."}]},
        ])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url, hide_thinking=False))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 200, raw)
        message = json.loads(raw)["choices"][0]["message"]
        self.assertTrue(message["reasoning_content"].startswith(prefix))
        self.assertIn(summary, message["reasoning_content"])
        self.assertNotIn("reconsider", message["reasoning_content"])
        self.assertEqual(message["content"], "The answer is 281.")
        self.assertEqual(len(requests), 3)
        summarizer = requests[1]
        self.assertEqual(summarizer["max_tokens"], 192)
        self.assertEqual(summarizer["reasoning_effort"], "none")
        self.assertNotIn("tools", summarizer)
        self.assertNotIn(prefix.strip(), summarizer["messages"][-1]["content"])
        self.assertEqual(requests[2]["continue_final_message"], "reasoning_content")

    def test_buffered_stream_respects_visibility_and_preserves_tool_calls(self):
        for hide in (True, False):
            with self.subTest(hide=hide):
                upstream, _ = fake_upstream([{"deltas": [
                    {"reasoning_content": "Private reasoning."},
                    {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                      "function": {"name": "lookup", "arguments": '{"q":'}}]},
                    {"tool_calls": [{"index": 0, "function": {"arguments": '"test"}'}}]},
                ], "finish": "tool_calls"}])
                with running(upstream) as upstream_url:
                    with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url, hide_thinking=hide))) as url:
                        status, raw, headers = chat(url, stream=True, stream_options={"include_usage": True})
                self.assertEqual(status, 200, raw)
                self.assertIn("text/event-stream", headers["Content-Type"])
                self.assertEqual(raw.count(b"data: [DONE]"), 1)
                frames = [json.loads(line[6:]) for line in raw.decode().splitlines()
                          if line.startswith("data: ") and line != "data: [DONE]"]
                deltas = [c["delta"] for f in frames for c in f["choices"]]
                tools = [d["tool_calls"][0] for d in deltas if "tool_calls" in d]
                self.assertEqual(tools[0]["function"], {"name": "lookup", "arguments": '{"q":"test"}'})
                self.assertEqual(b"Private reasoning" in raw, not hide)
                self.assertEqual(frames[-1]["usage"]["completion_tokens"], 20)

    def test_echoed_continuation_does_not_duplicate_the_preserved_prefix(self):
        prefix = "We have established that the answer is 281.\n"
        unit = "Let me consider the next step again. "
        upstream, _ = fake_upstream([
            {"deltas": [{"content": prefix}, {"content": unit * 12}]},
            {"echo": True, "deltas": [{"content": "Here is the proof."}]},
        ])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 200, raw)
        content = json.loads(raw)["choices"][0]["message"]["content"]
        self.assertEqual(content, prefix + unit + "Here is the proof.")
        self.assertEqual(content.count(prefix), 1)
        self.assertEqual(content.count(unit), 1)

    def test_budget_exhausted_empty_answer_is_reported_not_silently_returned(self):
        # Deliberately varied: an exactly repeating filler would trip the loop detector first.
        filler = " ".join(
            "Consideration number %d reaches a different conclusion about the matter at hand." % i
            for i in range(1, 12))
        upstream, _ = fake_upstream([{"deltas": [{"reasoning_content": filler}], "finish": "length"}])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(raw)["error"]["code"], "empty_answer_length")
        self.assertNotIn(b"Consideration", raw)

    def test_empty_content_with_a_tool_call_is_not_an_error(self):
        upstream, _ = fake_upstream([{"deltas": [
            {"tool_calls": [{"index": 0, "id": "c1", "type": "function",
                              "function": {"name": "f", "arguments": "{}"}}]},
        ], "finish": "tool_calls"}])
        with running(upstream) as upstream_url:
            with running(make_server(("127.0.0.1", 0), Config(upstream=upstream_url))) as url:
                status, raw, _ = chat(url)
        self.assertEqual(status, 200, raw)
        self.assertEqual(json.loads(raw)["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "f")

    def test_hide_thinking_preserves_answer_and_does_not_disable_reasoning(self):
        upstream, requests = fake_upstream([{"deltas": [
            {"role": "assistant", "reasoning_content": "Private reasoning."},
            {"content": "The answer is 281."},
        ]}])
        with running(upstream) as upstream_url:
            server = make_server(("127.0.0.1", 0), Config(upstream=upstream_url, hide_thinking=True))
            with running(server) as url:
                status, raw, _ = chat(url, reasoning_effort="high", temperature=0)
        self.assertEqual(status, 200, raw)
        message = json.loads(raw)["choices"][0]["message"]
        self.assertEqual(message["content"], "The answer is 281.")
        self.assertNotIn("reasoning_content", message)
        self.assertNotIn(b"Private reasoning", raw)
        self.assertEqual(requests[0]["reasoning_effort"], "high")
        self.assertEqual(requests[0]["temperature"], 0)
        self.assertEqual(requests[0]["reasoning_format"], "deepseek")
        self.assertTrue(requests[0]["stream"])


if __name__ == "__main__":
    unittest.main()
