"""OpenAI-compatible API — serve a model on a GPU and query it at localhost:8001.

The compute provisions one on-demand GPU and runs vLLM's OpenAI-compatible server
on port 8000. ``sky.Port`` bridges that to ``127.0.0.1:8001`` over SSH, so a local
OpenAI client talks to the remote model as if it were running next door.
"""

import time

from openai import OpenAI

import skyward as sky

MODEL = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


@sky.function
def serve() -> None:
    """Run vLLM's OpenAI-compatible server on the node (blocks forever)."""
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MODEL, "--host", "0.0.0.0", "--port", "8000",
         "--max-model-len", "32768"],
        check=True,
    )


def wait_until_ready(client: OpenAI, timeout: float = 600.0) -> None:
    """Poll the local endpoint until the model finishes loading."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client.models.list()
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError("model server did not become ready in time")


if __name__ == "__main__":
    with sky.Compute(
        provider=sky.RunPod(),
        accelerator=sky.accelerators.A40(),
        allocation="on_demand",
        nodes=sky.Nodes(min=0, initial=1, max=1),
        image=sky.Image(
            pip=["vllm"],
            apt=["gcc"],
            env={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
        ),
        ports=[sky.Port(remote=8000, local=8001)],
    ) as compute:
        server = serve() > compute

        client = OpenAI(base_url="http://localhost:8001/v1", api_key="skyward")
        wait_until_ready(client)

        reply = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "In one sentence, what is Skyward?"}],
        )
        print(reply.choices[0].message.content)
