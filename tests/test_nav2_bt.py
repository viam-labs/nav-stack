from pathlib import Path

import pytest

from src.config import Nav2Config
from src.models.nav_core import _tune_nav2_bt_xml, _write_nav2_bt_xml


_SAMPLE_BT = """
<root main_tree_to_execute="MainTree">
  <BehaviorTree ID="MainTree">
    <RecoveryNode number_of_retries="6" name="NavigateRecovery">
      <PipelineSequence name="NavigateWithReplanning">
        <RateController hz="1.0">
          <ComputePathToPose goal="{goal}" path="{path}" planner_id="GridBased"/>
        </RateController>
        <FollowPath path="{path}" controller_id="FollowPath"/>
      </PipelineSequence>
      <ReactiveFallback name="RecoveryFallback">
        <GoalUpdated/>
        <RoundRobin name="RecoveryActions">
          <Spin spin_dist="1.57"/>
          <Wait wait_duration="5"/>
          <BackUp backup_dist="0.30" backup_speed="0.05"/>
        </RoundRobin>
      </ReactiveFallback>
    </RecoveryNode>
  </BehaviorTree>
</root>
"""


def test_tune_nav2_bt_xml_rewrites_replan_and_recovery():
    out = _tune_nav2_bt_xml(
        _SAMPLE_BT,
        replan_hz=2.5,
        navigate_recovery_retries=3,
        recovery_wait_duration=2.0,
    )
    assert 'RateController hz="2.5"' in out
    assert 'RateController hz="1.0"' not in out
    assert 'RecoveryNode number_of_retries="3" name="NavigateRecovery"' in out
    assert 'Wait wait_duration="2.0"' in out
    assert 'Wait wait_duration="5"' not in out


def test_write_nav2_bt_xml_uses_config_defaults(tmp_path: Path):
    path = _write_nav2_bt_xml(tmp_path, Nav2Config())
    text = path.read_text(encoding="utf-8")
    assert 'RateController hz="2.0"' in text
    assert 'number_of_retries="4" name="NavigateRecovery"' in text
    assert 'Wait wait_duration="2.0"' in text


def test_nav2_config_progress_and_replan_defaults():
    cfg = Nav2Config.from_dict({})
    assert cfg.replan_frequency == 2.0
    assert cfg.progress_movement_time_allowance == 10.0
    assert cfg.navigate_recovery_retries == 4
    assert cfg.recovery_wait_duration == 2.0
    overrides = cfg.to_override_dict()
    assert overrides["movement_time_allowance"] == 10.0
    assert "replan_frequency" not in overrides


def test_nav2_config_overrides():
    cfg = Nav2Config.from_dict(
        {
            "replan_frequency": 3.0,
            "progress_movement_time_allowance": 8.0,
            "navigate_recovery_retries": 2,
            "recovery_wait_duration": 1.5,
        }
    )
    assert cfg.replan_frequency == 3.0
    assert cfg.progress_movement_time_allowance == 8.0
    assert cfg.navigate_recovery_retries == 2
    assert cfg.recovery_wait_duration == 1.5


def test_nav2_template_progress_allowance_matches_default():
    yaml = pytest.importorskip("yaml")
    params_file = Path(__file__).resolve().parents[1] / "params" / "nav2_params.yaml"
    data = yaml.safe_load(params_file.read_text(encoding="utf-8"))
    allowance = data["controller_server"]["ros__parameters"]["progress_checker"][
        "movement_time_allowance"
    ]
    assert allowance == 10.0
