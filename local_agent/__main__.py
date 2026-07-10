from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path.cwd().resolve()
MAX_READ_BYTES = 20_000
MAX_WRITE_BYTES = 50_000


SYSTEM_PROMPT = f"""
You are a helpful local coding agent running in a terminal.

You can use tools to inspect and update files, but you must stay inside this
project folder:

{PROJECT_ROOT}

Before writing files, briefly explain what you are changing. Prefer small,
clear edits. If a task is risky or ambiguous, ask a short clarifying question.
Only use a file tool when the user asks about project files or asks you to edit
the project. For greetings and general questions, answer directly. When you
need a file tool, return a JSON object with exactly this shape:
{{"name":"tool_name","arguments":{{...}}}}
""".strip()


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files under a project-relative directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Project-relative directory path. Use '.' for the project root.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the project.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Project-relative file path.",
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text to a project file, creating parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Project-relative file path.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete file contents to write.",
                    },
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
]


def resolve_project_path(path: str) -> Path:
    target = (PROJECT_ROOT / path).resolve()
    if target != PROJECT_ROOT and PROJECT_ROOT not in target.parents:
        raise ValueError(f"Path escapes project root: {path}")
    return target


def list_files(path: str) -> str:
    target = resolve_project_path(path)
    if not target.exists():
        return f"Directory does not exist: {path}"
    if not target.is_dir():
        return f"Not a directory: {path}"

    files = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        suffix = "/" if child.is_dir() else ""
        files.append(f"{child.relative_to(PROJECT_ROOT)}{suffix}")
    return "\n".join(files) if files else "Directory is empty."


def read_file(path: str) -> str:
    target = resolve_project_path(path)
    if not target.exists():
        return f"File does not exist: {path}"
    if not target.is_file():
        return f"Not a file: {path}"

    data = target.read_bytes()
    if len(data) > MAX_READ_BYTES:
        data = data[:MAX_READ_BYTES]
        note = f"\n\n[truncated after {MAX_READ_BYTES} bytes]"
    else:
        note = ""

    return data.decode("utf-8", errors="replace") + note


def write_file(path: str, content: str) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) > MAX_WRITE_BYTES:
        return f"Refusing to write more than {MAX_WRITE_BYTES} bytes."

    target = resolve_project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Wrote {len(encoded)} bytes to {target.relative_to(PROJECT_ROOT)}"


def call_tool(name: str, arguments: dict[str, Any]) -> str:
    try:
        if name == "list_files":
            return list_files(arguments["path"])
        if name == "read_file":
            return read_file(arguments["path"])
        if name == "write_file":
            return write_file(arguments["path"], arguments["content"])
        return f"Unknown tool: {name}"
    except Exception as exc:
        return f"Tool error: {exc}"


def ask_ollama(
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = TOOLS,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "tools": tools or [],
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Install it from https://ollama.com/download "
            "and make sure Ollama is running."
        ) from exc
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc


def extract_text_tool_call(content: str) -> dict[str, Any] | None:
    """Accept JSON tool calls from models that do not emit native tool calls."""
    candidates = [content.strip()]
    if content.strip().startswith("```"):
        candidates.append(content.strip().strip("`").removeprefix("json").strip())

    start = content.find("{")
    end = content.rfind("}")
    if start >= 0 and end > start:
        candidates.append(content[start : end + 1])

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(value, dict)
            and value.get("name") in {tool["function"]["name"] for tool in TOOLS}
            and isinstance(value.get("arguments"), dict)
        ):
            return value
    return None


def direct_file_request(user_text: str) -> tuple[str, str] | None:
    lowered = user_text.lower()
    if "list files" in lowered or "show files" in lowered:
        return "list", "."

    match = re.search(r"\bread\s+[`\"]?([^`\"\s]+)", user_text, re.IGNORECASE)
    if match:
        return "read", match.group(1)
    return None


def run_agent(base_url: str, model: str, user_text: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    direct_request = direct_file_request(user_text)
    if direct_request:
        action, path = direct_request
        if action == "list":
            result = list_files(path)
            print(f"\nAgent: Files in the project:\n{result}\n")
            return [*history, {"role": "user", "content": user_text}, {"role": "assistant", "content": result}]

        file_content = read_file(path)
        summary_messages = [
            {"role": "system", "content": "Summarize the supplied file clearly and briefly. Do not use tools."},
            {
                "role": "user",
                "content": f"Summarize {path}:\n\n--- file content ---\n{file_content}\n--- end file ---",
            },
        ]
        try:
            response = ask_ollama(base_url, model, summary_messages, tools=[])
            summary = response.get("message", {}).get("content", "")
        except RuntimeError as exc:
            print(f"\nAgent: {exc}\n")
            return history
        print(f"\nAgent: {summary}\n")
        return [*history, {"role": "user", "content": user_text}, {"role": "assistant", "content": summary}]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user_text},
    ]

    for _ in range(8):
        try:
            response = ask_ollama(base_url, model, messages)
        except RuntimeError as exc:
            print(f"\nAgent: {exc}\n")
            return messages

        assistant_message = response.get("message", {})
        messages.append(assistant_message)
        tool_calls = assistant_message.get("tool_calls", [])
        if not tool_calls:
            text_tool_call = extract_text_tool_call(assistant_message.get("content", ""))
            if text_tool_call:
                tool_calls = [{"function": text_tool_call}]
        if not tool_calls:
            print(f"\nAgent: {assistant_message.get('content', '')}\n")
            return messages

        for tool_call in tool_calls:
            function = tool_call.get("function", {})
            args = function.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args or "{}")
            result = call_tool(function.get("name", ""), args)
            messages.append(
                {
                    "role": "tool",
                    "name": function.get("name", ""),
                    "content": result,
                }
            )

    print("\nAgent: I stopped because the tool loop ran too many times.\n")
    return messages


def main() -> None:
    model = os.getenv("LOCAL_MODEL", "qwen2.5-coder:3b")
    base_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    history: list[dict[str, Any]] = []

    print(f"Local Agent is ready using {model}. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if user_text.lower() in {"exit", "quit"}:
            print("Bye.")
            return
        if not user_text:
            continue

        history = run_agent(base_url, model, user_text, history)


if __name__ == "__main__":
    main()
