"""Accelerator types and factory functions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type H100MIG = Literal["1g.10gb", "1g.20gb", "2g.20gb", "3g.40gb", "4g.40gb", "7g.80gb"]
type A100MIG = Literal["1g.5gb", "1g.10gb", "2g.10gb", "2g.20gb", "3g.20gb", "3g.40gb", "4g.20gb", "4g.40gb", "7g.40gb", "7g.80gb"]


@dataclass(frozen=True)
class Accelerator:
    accelerator: str
    memory: str
    count: int = 1
    multiple_instance: str | list[str] | None = None

    class NVIDIA:
        @staticmethod
        def H100(
            count: int = 1,
            memory: Literal["40GB", "80GB"] = "80GB",
            form_factor: Literal["SXM", "PCIe", "NVL"] | None = None,
            mig: H100MIG | list[H100MIG] | None = None,
        ) -> Accelerator:
            if form_factor:
                name = f"H100-{form_factor}"
                mem = "94GB" if form_factor == "NVL" else memory
            else:
                name = "H100"
                mem = memory
            return Accelerator(name, mem, count, mig)

        @staticmethod
        def H200(count: int = 1) -> Accelerator:
            return Accelerator("H200", "141GB", count)

        @staticmethod
        def A100(
            count: int = 1,
            memory: Literal["40GB", "80GB"] = "80GB",
            mig: A100MIG | list[A100MIG] | None = None,
        ) -> Accelerator:
            return Accelerator("A100", memory, count, mig)

        @staticmethod
        def B100(count: int = 1) -> Accelerator:
            return Accelerator("B100", "192GB", count)

        @staticmethod
        def B200(count: int = 1) -> Accelerator:
            return Accelerator("B200", "192GB", count)

        @staticmethod
        def GB200(count: int = 1) -> Accelerator:
            return Accelerator("GB200", "384GB", count)

        @staticmethod
        def GH200(count: int = 1) -> Accelerator:
            return Accelerator("GH200", "96GB", count)

        @staticmethod
        def L4(count: int = 1) -> Accelerator:
            return Accelerator("L4", "24GB", count)

        @staticmethod
        def L40(count: int = 1) -> Accelerator:
            return Accelerator("L40", "48GB", count)

        @staticmethod
        def L40S(count: int = 1) -> Accelerator:
            return Accelerator("L40S", "48GB", count)

        @staticmethod
        def T4(count: int = 1) -> Accelerator:
            return Accelerator("T4", "16GB", count)

        @staticmethod
        def A10(count: int = 1) -> Accelerator:
            return Accelerator("A10", "24GB", count)

        @staticmethod
        def A10G(count: int = 1) -> Accelerator:
            return Accelerator("A10G", "24GB", count)

        @staticmethod
        def A2(count: int = 1) -> Accelerator:
            return Accelerator("A2", "16GB", count)

        @staticmethod
        def V100(count: int = 1, memory: Literal["16GB", "32GB"] = "32GB") -> Accelerator:
            return Accelerator("V100", memory, count)

        @staticmethod
        def P100(count: int = 1) -> Accelerator:
            return Accelerator("P100", "16GB", count)

        @staticmethod
        def P4(count: int = 1) -> Accelerator:
            return Accelerator("P4", "8GB", count)

        @staticmethod
        def K80(count: int = 1) -> Accelerator:
            return Accelerator("K80", "12GB", count)

        @staticmethod
        def RTX3080(count: int = 1) -> Accelerator:
            return Accelerator("RTX3080", "10GB", count)

        @staticmethod
        def RTX3090(count: int = 1) -> Accelerator:
            return Accelerator("RTX3090", "24GB", count)

        @staticmethod
        def RTX4080(count: int = 1) -> Accelerator:
            return Accelerator("RTX4080", "16GB", count)

        @staticmethod
        def RTX4090(count: int = 1) -> Accelerator:
            return Accelerator("RTX4090", "24GB", count)

    class AMD:
        @staticmethod
        def MI(
            model: Literal["50", "100", "210", "250", "250X", "300A", "300B", "300X"],
            count: int = 1,
        ) -> Accelerator:
            memory = {
                "50": "16GB", "100": "32GB", "210": "64GB",
                "250": "128GB", "250X": "128GB",
                "300A": "128GB", "300B": "192GB", "300X": "192GB",
            }
            return Accelerator(f"MI{model}", memory[model], count)

    class Habana:
        @staticmethod
        def Gaudi(version: Literal[1, 2, 3] = 3, count: int = 1) -> Accelerator:
            memory = {1: "32GB", 2: "96GB", 3: "128GB"}
            name = "Gaudi" if version == 1 else f"Gaudi{version}"
            return Accelerator(name, memory[version], count)

    class AWS:
        @staticmethod
        def Trainium(version: Literal[1, 2, 3] = 2, count: int = 1) -> Accelerator:
            memory = {1: "32GB", 2: "64GB", 3: "128GB"}
            return Accelerator(f"Trainium{version}", memory[version], count)

        @staticmethod
        def Inferentia(version: Literal[1, 2] = 2, count: int = 1) -> Accelerator:
            memory = {1: "8GB", 2: "32GB"}
            return Accelerator(f"Inferentia{version}", memory[version], count)

    class Google:
        @staticmethod
        def TPU(
            version: Literal["v2", "v3", "v4", "v5e", "v5p", "v6"] = "v5p",
            count: int = 1,
        ) -> Accelerator:
            memory = {"v2": "8GB", "v3": "16GB", "v4": "32GB", "v5e": "16GB", "v5p": "95GB", "v6": "32GB"}
            return Accelerator(f"TPU{version}", memory[version], count)

        @staticmethod
        def TPUSlice(
            version: Literal["v2-8", "v3-8", "v3-32", "v4-64", "v5e-4", "v5p-8"],
        ) -> Accelerator:
            config = {
                "v2-8": ("64GB", 8),
                "v3-8": ("128GB", 8),
                "v3-32": ("512GB", 32),
                "v4-64": ("2TB", 64),
                "v5e-4": ("64GB", 4),
                "v5p-8": ("760GB", 8),
            }
            mem, cnt = config[version]
            return Accelerator(f"TPU{version}", mem, cnt)
