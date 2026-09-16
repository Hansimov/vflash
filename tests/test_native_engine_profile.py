from vflash.native.h3_native_engine import H3NativeEngine


class _Event:
    def __init__(self, milliseconds):
        self.milliseconds = milliseconds

    def elapsed_time(self, other):
        return other.milliseconds - self.milliseconds


def test_engine_profile_materializes_primary_stream_and_host_sections():
    engine = H3NativeEngine.__new__(H3NativeEngine)
    boundaries = {
        "start": 0,
        "row_setup_end": 1,
        "input_pack_end": 3,
        "invocation_prepare_end": 6,
        "denoiser_end": 16,
        "final_layer_end": 20,
        "latent_update_end": 25,
    }
    host = {
        "row_setup": 0.01,
        "input_pack": 0.02,
        "invocation_prepare": 0.03,
        "denoiser": 0.04,
        "final_layer": 0.05,
        "latent_update": 0.06,
        "engine_step": 0.21,
    }
    engine._denoise_profile_records = [
        {
            "evaluation_index": 7,
            "events": {name: _Event(value) for name, value in boundaries.items()},
            "host_seconds": host,
        }
    ]

    profile = engine.finish_denoise_profile()

    evaluation = profile["evaluations"][0]
    assert evaluation["evaluation_index"] == 7
    assert evaluation["primary_gpu_seconds"] == {
        "row_setup": 0.001,
        "input_pack": 0.002,
        "invocation_prepare": 0.003,
        "denoiser": 0.01,
        "final_layer": 0.004,
        "latent_update": 0.005,
        "primary_stream": 0.025,
    }
    assert profile["host_seconds"] == host
    assert engine._denoise_profile_records is None
