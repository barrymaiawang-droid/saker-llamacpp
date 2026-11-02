"""Replay the user's project task using a native file-write tool, in scratch only."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import urllib.request
from probe_engine import call, server


def generate(base, body, out):
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    start = time.perf_counter()
    first = None
    message = {"role":"assistant", "content":"", "reasoning_content":"", "tool_calls":[]}
    final = {}
    last_progress = 0
    with urllib.request.urlopen(request, timeout=1200) as response, (out / "stream.jsonl").open("w") as stream:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            data = line[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            stream.write(json.dumps(chunk)+"\n")
            if chunk.get("usage"):
                final["usage"] = chunk["usage"]
            if chunk.get("timings"):
                final["timings"] = chunk["timings"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if delta and first is None:
                    first = time.perf_counter()-start
                for key in ("content", "reasoning_content"):
                    message[key] += delta.get(key) or ""
                for tc in delta.get("tool_calls", []):
                    i = tc["index"]
                    while len(message["tool_calls"]) <= i:
                        message["tool_calls"].append({"id":"", "type":"function", "function":{"name":"", "arguments":""}})
                    target = message["tool_calls"][i]
                    if tc.get("id"):
                        target["id"] = tc["id"]
                    fn = tc.get("function", {})
                    for key in ("name", "arguments"):
                        target["function"][key] += fn.get(key) or ""
                if choice.get("finish_reason"):
                    final["finish_reason"] = choice["finish_reason"]
            if time.perf_counter()-last_progress > 3:
                progress = {"elapsed":time.perf_counter()-start, "reasoning_chars":len(message["reasoning_content"]),
                            "tool_chars":sum(len(t["function"]["arguments"]) for t in message["tool_calls"])}
                (out / "progress.json").write_text(json.dumps(progress)+"\n")
                last_progress = time.perf_counter()
    final.update(message=message, first_delta_seconds=first, elapsed=time.perf_counter()-start)
    (out / "response.json").write_text(json.dumps(final,indent=2)+"\n")
    if final.get("finish_reason") != "tool_calls" or len(message["tool_calls"]) != 1:
        raise RuntimeError("project did not finish with one complete file call: " + str(final.get("finish_reason")))
    fn = message["tool_calls"][0]["function"]
    if fn["name"] != "write_file":
        raise RuntimeError("unexpected tool " + fn["name"])
    content = json.loads(fn["arguments"])["content"]
    if "</html>" not in content.lower():
        raise RuntimeError("incomplete HTML")
    (out / "pelican.html").write_text(content)
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["vulkan","hip","cpu"], default="vulkan",
                   help="cpu runs the reference implementation with no GPU work; expect well under 5 tokens/s")
    p.add_argument("--profile", choices=["ar","mtp","shortlist"], default="ar")
    p.add_argument("--legacy-sampling", action="store_true")
    p.add_argument("--wide-warp", action="store_true")
    p.add_argument("--effort", choices=["low","medium","xhigh"], default="xhigh")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=12288)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--threads", type=int, default=None, help="default 4 for GPU arms, 16 for the cpu arm")
    p.add_argument("--port", type=int, default=18153)
    args = p.parse_args()
    if args.backend == "cpu" and args.profile != "ar":
        p.error("the cpu arm is the plain reference implementation; use --profile ar")
    threads = args.threads if args.threads is not None else (16 if args.backend == "cpu" else 4)
    launch = SimpleNamespace(backend=args.backend,profile=args.profile,ctx=args.ctx,ubatch=256,threads=threads,port=args.port,no_retune=False,wide_warp=args.wide_warp,out=args.out)
    tools = [{"type":"function", "function":{"name":"write_file", "description":"Write the complete standalone HTML webpage.",
             "parameters":{"type":"object", "properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}]
    body = {"model":"q38gsq-review", "messages":[
        {"role":"system", "content":"You are building the requested webpage. Create the complete page in one write_file call. Use Three.js 0.160.0 with its ES module import map and matching OrbitControls if you need it. No other files are available. Write pelican.html."},
        {"role":"user", "content":"make a peligan rides a bike 3d webpage, dont search or refre to any other webpage in the folder, make one independantly on your own"}],
        "tools":tools,"tool_choice":"auto","max_tokens":args.tokens,"seed":args.seed,"stream":True,"stream_options":{"include_usage":True},
        "temperature":.6 if args.legacy_sampling else 1.,"top_p":.95,"top_k":64 if args.legacy_sampling else 20,"min_p":.05 if args.legacy_sampling else 0.,
        "chat_template_kwargs":{"enable_thinking":True,"reasoning_effort":args.effort,"preserve_thinking":True}}
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out / "request.json").write_text(json.dumps(body,indent=2)+"\n")
    with server(launch) as (base, metadata):
        result = generate(base,body,args.out)
        (args.out / "metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k!="message"}),flush=True)
        print("HTML",args.out / "pelican.html",flush=True)


if __name__ == "__main__":
    main()
