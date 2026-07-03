"""Example: autonomous browser computer-use task.

Usage:  python examples/browse_task.py "Go to en.wikipedia.org and find the population of Iceland"
(Requires the vLLM server to be running — see scripts/serve_vllm.sh)
"""

import sys

from visual_agent.agent import run_browse_task

task = " ".join(sys.argv[1:]) or "Go to https://example.com and describe what the page says."

for event in run_browse_task(task):
    if event.kind == "thought":
        print(f"[thought] {event.text}")
    elif event.kind == "tool":
        print(f"[action]  {event.tool_name}({event.tool_args})")
    elif event.kind == "screenshot":
        print(f"[shot]    {event.screenshot_path}")
    elif event.kind == "final":
        print(f"\n[result]  {event.text}")
