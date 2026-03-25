"""Progress Bars — tqdm updates in-place on the console.

Demonstrates that progress bars from libraries like tqdm render
aesthetically in Skyward's console: lines overwrite in-place instead
of flooding the log with one line per update.

    ┌──────────────────────────────────────┐
    │  instance1  Epoch 3/5: 75%|███▊ |   │
    │                                      │
    │  (updates in-place, no line spam)    │
    └──────────────────────────────────────┘
"""

from time import sleep

import skyward as sky


@sky.function
def train_with_progress(epochs: int, steps: int) -> dict:
    """Simulate training with tqdm progress bars."""
    import time

    from tqdm import tqdm

    print("Starting training...")
    results = {}
    for epoch in range(1, epochs + 1):
        print(f"Beginning epoch {epoch}/{epochs}")
        total_loss = 0.0
        bar = tqdm(range(steps), desc=f"Epoch {epoch}/{epochs}", ncols=80)
        for step in bar:
            loss = 1.0 / (step + 1 + epoch)
            total_loss += loss
            bar.set_postfix(loss=f"{loss:.4f}")
            time.sleep(0.2)

        avg_loss = total_loss / steps
        results[f"epoch_{epoch}"] = avg_loss
        print(f"Epoch {epoch} complete — avg loss: {avg_loss:.4f}")

    print("All epochs done!")
    return results


if __name__ == "__main__":
    with sky.Compute(
        provider=sky.Container(),
        image=sky.Image(pip=["tqdm"]),
        nodes=3,
    ) as pool:
        results = train_with_progress(epochs=2, steps=25) @ pool
        sleep(1)
        for i, r in enumerate(results):
            print(f"\nNode {i}: {r}")
