from src.nav.motion_summary import describe_cmd_vel, summarize_nav_motion


def test_describe_hard_right_turn_with_forward():
    last = {
        "ros_vx_mps": 0.4,
        "ros_vy_mps": 0.0,
        "ros_vtheta_rad_s": -1.0,
        "source": "nav2",
        "age_s": 0.1,
    }
    phrase = describe_cmd_vel(
        last, max_vel_x=0.75, max_vel_theta=1.2, held_s=3.0
    )
    assert "forward" in phrase
    assert "hard" in phrase
    assert "right" in phrase
    assert "3.0" in phrase


def test_describe_spin_in_place_left():
    phrase = describe_cmd_vel(
        {"ros_vx_mps": 0.0, "ros_vy_mps": 0.0, "ros_vtheta_rad_s": 0.5, "source": "nav2"},
        max_vel_x=0.75,
        max_vel_theta=1.2,
    )
    assert "spinning" in phrase
    assert "left" in phrase


def test_summarize_idle_stopped():
    out = summarize_nav_motion(
        {
            "active": False,
            "state": "idle",
            "last_cmd_vel": {
                "ros_vx_mps": 0.0,
                "ros_vy_mps": 0.0,
                "ros_vtheta_rad_s": 0.0,
                "source": "stop",
                "age_s": 2.0,
            },
            "cmd_vel_history": [],
        }
    )
    assert out["summary"].startswith("idle:")
    assert "stopped" in out["action"]


def test_summarize_nav2_with_distance_and_held():
    out = summarize_nav_motion(
        {
            "active": True,
            "state": "navigating",
            "distance_remaining": 1.25,
            "last_cmd_vel": {
                "ros_vx_mps": 0.5,
                "ros_vy_mps": 0.0,
                "ros_vtheta_rad_s": -0.9,
                "source": "nav2",
                "age_s": 0.05,
            },
            "cmd_vel_history": [
                {
                    "ros_vx_mps": 0.5,
                    "ros_vtheta_rad_s": 0.0,
                    "age_s": 3.1,
                },
                {
                    "ros_vx_mps": 0.5,
                    "ros_vtheta_rad_s": -0.9,
                    "age_s": 0.05,
                },
            ],
        },
        max_vel_x=0.75,
        max_vel_theta=1.2,
    )
    assert "Nav2 navigating" in out["summary"]
    assert "1.2 m remaining" in out["summary"]
    assert "hard" in out["action"] and "right" in out["action"]
    assert out["held_s"] == 3.1


def test_summarize_includes_goal_relative_and_progress():
    # Robot at origin facing +x; goal ahead-right; turning right while driving.
    out = summarize_nav_motion(
        {
            "active": True,
            "state": "active",
            "pose": {"x": 0.0, "y": 0.0, "theta": 0.0},
            "goal": {"x": 2.0, "y": -2.0, "theta": 0.0, "name": "dock"},
            "last_cmd_vel": {
                "ros_vx_mps": 0.4,
                "ros_vy_mps": 0.0,
                "ros_vtheta_rad_s": -0.5,
                "source": "nav2",
                "age_s": 0.1,
            },
            "cmd_vel_history": [],
        },
        max_vel_x=0.75,
        max_vel_theta=1.2,
    )
    assert "dock" in out["summary"]
    assert "ahead and to the right" in out["goal_relative"]
    assert out["toward_goal"] is not None
    assert "closing distance" in out["toward_goal"] or "steering toward" in out["toward_goal"]
    assert "—" in out["summary"]
