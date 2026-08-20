from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sol_overlay_is_pinned_on_top_of_the_sage_capable_image():
    dockerfile = (ROOT / "docker/Dockerfile.sol-attn").read_text()
    assert "ARG SGLANG_BASE_IMAGE=" in dockerfile
    assert "5fe5febdf0f59fee1c0b44a5ce6665df0dabd247" in dockerfile
    assert "#subdirectory=techniques/sparse_backends" in dockerfile
    assert "import sol_attn" in dockerfile


def test_optimization_toggle_reinstalls_every_partition():
    script = (ROOT / "scripts/configure_sol_ab.sh").read_text()
    assert "set_env OPTIMIZATION_STACK_ENABLED 1" in script
    assert "set_env OPTIMIZATION_STACK_ENABLED 0" in script
    assert "on every 4-H200 partition" in script
    assert "exec ./install.sh" in script


def test_sol_stack_verifier_fails_closed_on_all_three_optimizations():
    script = (ROOT / "scripts/verify_sol_stack.sh").read_text()
    assert "Using sol_attn attention backend" in script
    assert "Attention backends for audio_vae: fa" in script
    assert 'required_env QUANTIZATION "${SOL_QUANTIZATION}"' in script
    assert 'required_env LORA_MERGE_MODE "${SOL_LORA_MERGE_MODE}"' in script
    assert 'required_env SGLANG_CACHE_DIT_ENABLED "${SOL_CACHE_DIT_ENABLED}"' in script
    assert "f\"--quantization {os.environ['QUANTIZATION']}\"" in script
    assert "f\"--lora-merge-mode {os.environ['LORA_MERGE_MODE']}\"" in script
    assert "import cache_dit" in script
    assert "Fp8Config.get_name()" in script


def test_sol_toggle_wrappers_are_one_command_entrypoints():
    enable = (ROOT / "enable_sol_ab.sh").read_text()
    disable = (ROOT / "disable_sol_ab.sh").read_text()
    assert 'configure_sol_ab.sh" enable' in enable
    assert 'configure_sol_ab.sh" disable' in disable
