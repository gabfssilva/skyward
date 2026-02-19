from __future__ import annotations

import pytest

from skyward.providers.gcp.instances import (
    default_n1_for_gpus,
    estimate_vram,
    is_guest_attachable,
    match_accelerator_name,
    parse_builtin_gpu_count,
    select_image_family,
)

ZONE_ACCELS = [
    "nvidia-tesla-t4",
    "nvidia-tesla-v100",
    "nvidia-a100-80gb",
    "nvidia-h100-80gb",
    "nvidia-l4",
    "nvidia-tesla-p100",
]


class TestMatchAcceleratorName:
    def test_a100_match(self):
        result = match_accelerator_name("A100", ZONE_ACCELS)
        assert result == "nvidia-a100-80gb"

    def test_h100_match(self):
        result = match_accelerator_name("H100", ZONE_ACCELS)
        assert result == "nvidia-h100-80gb"

    def test_t4_match(self):
        result = match_accelerator_name("T4", ZONE_ACCELS)
        assert result == "nvidia-tesla-t4"

    def test_l4_match(self):
        result = match_accelerator_name("L4", ZONE_ACCELS)
        assert result == "nvidia-l4"

    def test_v100_match(self):
        result = match_accelerator_name("V100", ZONE_ACCELS)
        assert result == "nvidia-tesla-v100"

    def test_p100_match(self):
        result = match_accelerator_name("P100", ZONE_ACCELS)
        assert result == "nvidia-tesla-p100"

    def test_case_insensitive(self):
        result = match_accelerator_name("a100", ZONE_ACCELS)
        assert result == "nvidia-a100-80gb"

    def test_no_match_raises(self):
        with pytest.raises(RuntimeError, match="No GCP accelerator matches 'Z9000'"):
            match_accelerator_name("Z9000", ZONE_ACCELS)

    def test_no_match_empty_list_raises(self):
        with pytest.raises(RuntimeError, match="No GCP accelerator matches"):
            match_accelerator_name("A100", [])

    def test_fallback_normalized_match(self):
        result = match_accelerator_name("nvidia-tesla-t4", ZONE_ACCELS)
        assert result == "nvidia-tesla-t4"


class TestSelectImageFamily:
    def test_gpu_image(self):
        result = select_image_family(has_gpu=True)
        assert "deeplearning-platform-release" in result

    def test_cpu_image(self):
        result = select_image_family(has_gpu=False)
        assert "ubuntu-os-cloud" in result

    def test_gpu_has_cuda(self):
        result = select_image_family(has_gpu=True)
        assert "cu128" in result

    def test_cpu_has_ubuntu(self):
        result = select_image_family(has_gpu=False)
        assert "ubuntu-2404" in result


class TestEstimateVram:
    @pytest.mark.parametrize(
        ("accel_type", "expected_gb"),
        [
            ("nvidia-h200-141gb", 141),
            ("nvidia-h100-80gb", 80),
            ("nvidia-a100-80gb", 80),
            ("nvidia-a100-40gb", 40),
            ("nvidia-l4", 24),
            ("nvidia-l40", 48),
            ("nvidia-tesla-v100", 16),
            ("nvidia-tesla-t4", 16),
            ("nvidia-tesla-p100", 16),
            ("nvidia-tesla-p4", 8),
            ("unknown-gpu-xyz", 0),
        ],
    )
    def test_vram_estimates(self, accel_type, expected_gb):
        assert estimate_vram(accel_type) == expected_gb


class TestParseBuiltinGpuCount:
    def test_a2_highgpu_4g(self):
        assert parse_builtin_gpu_count("a2-highgpu-4g") == 4

    def test_a2_highgpu_1g(self):
        assert parse_builtin_gpu_count("a2-highgpu-1g") == 1

    def test_a2_highgpu_8g(self):
        assert parse_builtin_gpu_count("a2-highgpu-8g") == 8

    def test_g2_standard_no_suffix(self):
        assert parse_builtin_gpu_count("g2-standard-4") == 1

    def test_a3_mega_8g(self):
        assert parse_builtin_gpu_count("a3-mega-8g") == 8

    def test_no_match_defaults_to_1(self):
        assert parse_builtin_gpu_count("n1-standard-8") == 1


class TestIsGuestAttachable:
    def test_t4_is_guest(self):
        assert is_guest_attachable("nvidia-tesla-t4") is True

    def test_v100_is_guest(self):
        assert is_guest_attachable("nvidia-tesla-v100") is True

    def test_p100_is_guest(self):
        assert is_guest_attachable("nvidia-tesla-p100") is True

    def test_p4_is_guest(self):
        assert is_guest_attachable("nvidia-tesla-p4") is True

    def test_l4_not_guest(self):
        assert is_guest_attachable("nvidia-l4") is False

    def test_a100_not_guest(self):
        assert is_guest_attachable("nvidia-a100-80gb") is False

    def test_h100_not_guest(self):
        assert is_guest_attachable("nvidia-h100-80gb") is False


class TestDefaultN1ForGpus:
    def test_1_gpu(self):
        assert default_n1_for_gpus(1) == "n1-standard-8"

    def test_2_gpus(self):
        assert default_n1_for_gpus(2) == "n1-standard-16"

    def test_4_gpus(self):
        assert default_n1_for_gpus(4) == "n1-standard-32"

    def test_8_gpus(self):
        assert default_n1_for_gpus(8) == "n1-standard-96"

    def test_unknown_count_defaults(self):
        assert default_n1_for_gpus(16) == "n1-standard-8"
