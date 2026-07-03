"""Example: one-shot vision analysis of an image.

Usage:  python examples/analyze_chart.py path/to/chart.png
(Requires the vLLM server to be running — see scripts/serve_vllm.sh)
"""

import sys

from visual_agent.chat import build_user_message, stream_chat

image_path = sys.argv[1] if len(sys.argv) > 1 else None
if not image_path:
    sys.exit("usage: python examples/analyze_chart.py <image>")

message = build_user_message(
    "Describe this image in detail. If it is a chart or diagram, extract the "
    "key numbers and the main takeaway.",
    [image_path],
)
for token in stream_chat([message]):
    print(token, end="", flush=True)
print()
