from __future__ import annotations

from src.ros import pcshm


def test_pcshm_round_trip():
    name = "/viam-pc-navt"
    payload = b"xyz" * 1000
    w = pcshm.open_writer(name)
    try:
        r = pcshm.open_reader(name)
        try:
            w.write(payload, timestamp_ns=123)
            got, ts = r.read()
            assert got == payload
            assert ts == 123
            w.write(b"second")
            got, _ = r.read()
            assert got == b"second"
        finally:
            r.close()
    finally:
        w.close()


def test_pcshm_no_frame_before_write():
    name = "/viam-pc-navt2"
    w = pcshm.open_writer(name)
    try:
        r = pcshm.open_reader(name)
        try:
            try:
                r.read()
            except pcshm.NoFrameError:
                pass
            else:
                raise AssertionError("expected NoFrameError")
        finally:
            r.close()
    finally:
        w.close()
