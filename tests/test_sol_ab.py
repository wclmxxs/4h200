from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sol_overlay_is_pinned_on_top_of_the_sage_capable_image():
    dockerfile = (ROOT / "docker/Dockerfile.sol-attn").read_text()
    assert "ARG SGLANG_BASE_IMAGE=" in dockerfile
    assert "5fe5febdf0f59fee1c0b44a5ce6665df0dabd247" in dockerfile
    assert "#subdirectory=techniques/sparse_backends" in dockerfile
    assert "import sol_attn" in dockerfile


def test_sglang_base_image_is_pinned_to_patch_commit():
    dockerfile = (ROOT / "docker/Dockerfile.sglang").read_text()
    env = (ROOT / "config/env.example").read_text()

    assert "nightly-dev-20260812-c7c03ec5@sha256:d7538b2" in dockerfile
    assert "SGLANG_EXPECTED_COMMIT=c7c03ec53b" in dockerfile
    assert 'git rev-parse --short=10 HEAD' in dockerfile
    assert "SGLANG_BASE_IMAGE=lmsysorg/sglang:dev" not in env


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


def test_install_migrates_the_previous_balanced_defaults():
    script = (ROOT / "install.sh").read_text()
    assert (
        "migrate_env_default SOL_ATTENTION_BACKEND_CONFIG "
        "dense_backend=sage_attn,dense_steps=2,kv_splits=auto,tau=1.0 "
        "dense_backend=sage_attn,dense_steps=1,kv_splits=auto,tau=1.25"
    ) in script
    assert "migrate_env_default SOL_CACHE_DIT_WARMUP 2 1" in script
    assert "migrate_env_default SOL_CACHE_DIT_RDT 0.04 0.08" in script
    assert "migrate_env_default SOL_CACHE_DIT_MC 1 2" in script
    assert (
        "migrate_env_default SOL_ATTENTION_BACKEND_CONFIG "
        "dense_backend=sage_attn,dense_steps=1,kv_splits=auto,tau=1.25 "
        "dense_backend=sage_attn,dense_steps=0,kv_splits=auto,tau=1.5"
    ) in script
    assert "migrate_env_default SOL_CACHE_DIT_RDT 0.08 0.12" in script
    assert "migrate_env_default SOL_CACHE_DIT_MC 2 3" in script
    assert 'sglang_build_base_image} == "lmsysorg/sglang:dev"' in script
    assert "build_gpu_image docker/Dockerfile.sglang" in script
    assert "REBUILD_GPU_IMAGES=1 to rebuild" in script


def test_sol_toggle_wrappers_are_one_command_entrypoints():
    enable = (ROOT / "enable_sol_ab.sh").read_text()
    disable = (ROOT / "disable_sol_ab.sh").read_text()
    assert 'configure_sol_ab.sh" enable' in enable
    assert 'configure_sol_ab.sh" disable' in disable
