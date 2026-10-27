"""Authenticated text, SSE, function-call, concurrency and classification tests; no tool execution.

The profile's "checks" suite (inference.CHECK_SUITES) decides which of them run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


def final_number(content):
    """The last number in an answer, so "42", "42." and "The answer is 42" all count, and "43" does not."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", content.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, help="Portable profile JSON; defaults to the private run configuration")
    args = parser.parse_args()
    key = args.api_key_file.read_text().strip()
    base = args.base_url.rstrip("/")
    from inference import ROOT, check_suite, load
    profile = load(args.profile) if args.profile else load(ROOT / "RUN_CONFIG.json")["profile"]
    model = profile["served_model"]
    template = profile["client"].get("chat_template_kwargs", {})
    suite = check_suite(profile)
    results = {}

    def request(path, body=None, authenticated=True):
        headers = {"Content-Type": "application/json"}
        if authenticated:
            headers["Authorization"] = "Bearer " + key
        data = None if body is None else json.dumps(body).encode()
        return urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers), timeout=profile["client"]["timeout_seconds"])

    def completion(prompt, extra=None):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 1024, "chat_template_kwargs": template}
        body.update(extra or {})
        start = time.monotonic()
        with request("/v1/chat/completions", body) as response:
            obj = json.load(response)
        return {"elapsed_seconds": round(time.monotonic() - start, 3), "response": obj}

    def check(name, function):
        if name not in suite:
            return
        try:
            result = function()
            results[name] = {"passed": True, "result": result}
        except Exception as error:
            results[name] = {"passed": False, "error_type": type(error).__name__, "error": str(error).replace(key, "[REDACTED]")}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        # On failure, show why (the answer is in the error), so the job log alone explains it.
        print(name, "PASS" if results[name]["passed"] else "FAIL: " + results[name]["error"][:300], flush=True)

    def auth_check():
        try:
            with request("/v1/models", authenticated=False) as response:
                raise AssertionError(f"Unauthenticated request returned {response.status}")
        except urllib.error.HTTPError as error:
            assert error.code == 401, error.code
            return {"status": error.code}

    check("unauthenticated_rejected", auth_check)

    def models_check():
        with request("/v1/models") as response:
            obj = json.load(response)
        assert model in [item["id"] for item in obj["data"]]
        return obj

    check("model_catalog", models_check)

    def arithmetic():
        obj = completion("What is 17 + 25? Give only the number.")
        content = obj["response"]["choices"][0]["message"].get("content") or ""
        assert final_number(content) == 42, "answer: " + repr(content)
        return obj

    check("arithmetic", arithmetic)

    def english():
        obj = completion("In English, briefly greet the user and introduce yourself as an AI assistant running on an HPC system in one sentence.")
        content = obj["response"]["choices"][0]["message"].get("content") or ""
        assert content.strip(), "No final content"
        assert "hpc" in content.lower() or "high-performance computing" in content.lower(), content
        return obj

    check("english_response", english)

    def streaming():
        body = {"model": model, "messages": [{"role": "user", "content": "Say hello in one short sentence."}], "temperature": 0, "max_tokens": 1024, "stream": True, "chat_template_kwargs": template}
        start = time.monotonic()
        first = None
        content = []
        done = False
        events = 0
        with request("/v1/chat/completions", body) as response:
            assert response.headers.get("Content-Type", "").startswith("text/event-stream")
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                obj = json.loads(data)
                assert "error" not in obj, obj
                events += 1
                for choice in obj.get("choices", []):
                    text = choice.get("delta", {}).get("content") or ""
                    if text:
                        if first is None:
                            first = time.monotonic() - start
                        content.append(text)
        assert done and content, "No final content or terminal SSE event"
        return {"first_content_seconds": round(first, 3), "total_seconds": round(time.monotonic() - start, 3), "events": events, "content": "".join(content), "done": done}

    check("sse_streaming", streaming)

    def tool_call():
        tool = {"type": "function", "function": {"name": "sum_numbers", "description": "Add two integers.", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}}}
        obj = completion("Use sum_numbers to add 17 and 25.", {"tools": [tool], "tool_choice": {"type": "function", "function": {"name": "sum_numbers"}}})
        calls = obj["response"]["choices"][0]["message"].get("tool_calls") or []
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "sum_numbers"
        assert json.loads(calls[0]["function"]["arguments"]) == {"a": 17, "b": 25}
        return obj

    check("forced_function_call", tool_call)

    def concurrent():
        prompts = ["What is 17 + 25? Give only the number.", "What is 6 * 7? Give only the number.", "What is 50 - 8? Give only the number.", "What is 84 / 2? Give only the number."]
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(completion, prompts))
        answers = [obj["response"]["choices"][0]["message"].get("content") or "" for obj in responses]
        assert all(final_number(a) == 42 for a in answers), "answers: " + repr(answers)
        return {"batch_seconds": round(time.monotonic() - start, 3), "responses": responses}

    check("four_concurrent_requests", concurrent)

    def any_completion():
        obj = completion("Hello! Please reply with a short greeting.", {"max_tokens": 64})
        message = obj["response"]["choices"][0]["message"]
        assert any((message.get(field) or "").strip() for field in ("content", "reasoning_content", "reasoning")), message
        return obj

    check("any_completion", any_completion)

    def classification():
        body = {"model": model, "input": ["Hello, how are you today?"]}
        with request("/classify", body) as response:
            obj = json.load(response)
        assert obj["data"] and "label" in obj["data"][0], obj
        return obj

    check("classification", classification)
    raise SystemExit(0 if all(r["passed"] for r in results.values()) else 1)


if __name__ == "__main__":
    main()
