from __future__ import annotations

from core.runtime.core_runtime import _mlx_device


class _EnumStyleMlx:
    cpu = object()
    gpu = object()


class _CallableStyleMlx:
    @staticmethod
    def cpu():
        return "cpu-device"

    @staticmethod
    def gpu():
        return "gpu-device"


def test_mlx_device_accepts_enum_style_device_constants():
    assert _mlx_device(_EnumStyleMlx, "cpu") is _EnumStyleMlx.cpu
    assert _mlx_device(_EnumStyleMlx, "gpu") is _EnumStyleMlx.gpu


def test_mlx_device_accepts_callable_style_device_factories():
    assert _mlx_device(_CallableStyleMlx, "cpu") == "cpu-device"
    assert _mlx_device(_CallableStyleMlx, "gpu") == "gpu-device"
