from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_sol_overlay_is_pinned_and_keeps_baseline_image_separate():
    dockerfile = (ROOT / "docker/Dockerfile.sol-attn").read_text()
    assert "ARG SGLANG_BASE_IMAGE=" in dockerfile
    assert "5fe5febdf0f59fee1c0b44a5ce6665df0dabd247" in dockerfile
    assert "#subdirectory=techniques/sparse_backends" in dockerfile
    assert "import sol_attn" in dockerfile


def test_sol_toggle_targets_only_the_selected_nonzero_partition():
    script = (ROOT / "scripts/configure_sol_ab.sh").read_text()
    assert 'worker_service="h3-sglang-${SOL_AB_SLOT}"' in script
    assert 'up -d --no-deps --force-recreate "${worker_service}"' in script
    assert "h3-sglang-0" not in script
    assert "SOL_AB_SLOT <= 0" in script
    assert "dense_steps=2" in script
    assert "Using sol_attn attention backend" in script
    assert "set_env_default SOL_ATTN_STRICT 1" in script
    assert "set_env_default SOL_WARMUP_STEPS 3" in script


def test_sol_toggle_wrappers_are_one_command_entrypoints():
    enable = (ROOT / "enable_sol_ab.sh").read_text()
    disable = (ROOT / "disable_sol_ab.sh").read_text()
    assert 'configure_sol_ab.sh" enable' in enable
    assert 'configure_sol_ab.sh" disable' in disable
