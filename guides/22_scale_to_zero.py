"""Scale to Zero — let an idle pool reap to zero nodes, then wake on demand."""

import time

import skyward as sky


@sky.function
def train(epoch: int) -> dict:
    """Run one training step and report which node served it."""
    info = sky.instance_info()
    return {"epoch": epoch, "node": info.node}


if __name__ == "__main__":
    with sky.Compute(
        provider=sky.RunPod(),
        accelerator=sky.accelerators.RTX_3090(),
        nodes=sky.Nodes(desired=1, min=0),
        options=sky.Options(autoscale_idle_timeout=30.0, autoscale_cooldown=10.0),
    ) as compute:
        first = train(1) >> compute
        print(f"Epoch {first['epoch']} ran on node {first['node']}")

        print("Idling — pool reaps to zero...")
        while compute.current_nodes() > 0:
            time.sleep(5)
        print(f"Nodes now running: {compute.current_nodes()}")

        second = train(2) >> compute
        print(f"Epoch {second['epoch']} woke the pool on node {second['node']}")
