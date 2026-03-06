#!/usr/bin/env python3

"""File-based command transport for Python environments that cannot import rclpy."""

from __future__ import annotations

import json
import os
import tempfile
import time


DEFAULT_COMMAND_FILE = os.path.join(tempfile.gettempdir(), "sim2real_cmd.json")


class FileCommandPublisher:
    def __init__(self, path: str = DEFAULT_COMMAND_FILE) -> None:
        self.path = path
        self.seq = 0

    def _write(self, payload: dict) -> None:
        self.seq += 1
        data = {
            "seq": self.seq,
            "timestamp": time.time(),
            **payload,
        }
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp_path, self.path)

    def send_left_arm(self, values: list[float]) -> None:
        if len(values) != 7:
            raise ValueError(f"left arm expects 7 values, got {len(values)}")
        self._write({"left_arm": list(values)})

    def send_left_gripper(self, value: float) -> None:
        self._write({"left_gripper": float(value)})

    def send_right_arm(self, values: list[float]) -> None:
        if len(values) != 7:
            raise ValueError(f"right arm expects 7 values, got {len(values)}")
        self._write({"right_arm": list(values)})

    def send_right_hand(self, values: list[float]) -> None:
        if len(values) != 20:
            raise ValueError(f"right hand expects 20 values, got {len(values)}")
        self._write({"right_hand": list(values)})

    def send_left_full(self, arm: list[float], gripper: float) -> None:
        if len(arm) != 7:
            raise ValueError(f"left arm expects 7 values, got {len(arm)}")
        self._write({"left_arm": list(arm), "left_gripper": float(gripper)})

    def send_right_full(self, arm: list[float], hand: list[float]) -> None:
        if len(arm) != 7:
            raise ValueError(f"right arm expects 7 values, got {len(arm)}")
        if len(hand) != 20:
            raise ValueError(f"right hand expects 20 values, got {len(hand)}")
        self._write({"right_arm": list(arm), "right_hand": list(hand)})

    def spin_once(self) -> None:
        return

    def close(self) -> None:
        return


def create_file_publisher(path: str = DEFAULT_COMMAND_FILE) -> FileCommandPublisher:
    return FileCommandPublisher(path=path)
