#!/usr/bin/env python3
"""HTTP control plane for the Isaac-backed world-model navigation MVP.

This server is intentionally thin. It does not own Isaac, Nav2, or the world
model itself; it starts the Isaac episode runner with a concrete point goal,
tracks the process/report, and exposes a stable control/status API for WebUI.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_ENV = Path("/media/chatsign/data-002/gs-sdf/scripts/activate_isaac_nav_mvp_env.sh")
DEFAULT_VALIDATION_REPORT = Path(
    "/media/chatsign/data-002/isaac/nav-mvp/reports/fast_livo2_minimal_agent_validation.json"
)
DEFAULT_OUTPUT_DIR = Path("/media/chatsign/data-002/isaac/nav-mvp/reports")
EPISODE_HISTORY_CANDIDATES = [
    DEFAULT_OUTPUT_DIR / "fast_livo2_nav_episodes_go2.json",
    DEFAULT_OUTPUT_DIR / "fast_livo2_nav_episodes_go2_cuda_smoke_20260423.json",
    DEFAULT_OUTPUT_DIR / "fast_livo2_nav_episodes.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8892)
    parser.add_argument("--isaac-env", default=str(DEFAULT_ISAAC_ENV))
    parser.add_argument("--validation-report", default=str(DEFAULT_VALIDATION_REPORT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8890")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--robot-model", choices=("unitree-go2", "capsule"), default="unitree-go2")
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--goal-tolerance", type=float, default=0.6)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--bridge-publish-rate-hz", type=float, default=8.0)
    parser.add_argument("--viewer", action="store_true", help="Launch Isaac episodes with GUI rendering instead of --headless.")
    parser.add_argument("--step-sleep", type=float, default=0.0, help="Viewer-mode wall sleep per sim step.")
    parser.add_argument("--episode-hold-sec", type=float, default=0.0, help="Viewer-mode hold after each episode.")
    parser.add_argument("--hold-open-sec", type=float, default=0.0, help="Viewer-mode hold after the final episode.")
    return parser.parse_args()


def now_ms() -> int:
    return int(time.time() * 1000.0)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_response(
    handler: BaseHTTPRequestHandler,
    payload: dict[str, Any],
    status: HTTPStatus = HTTPStatus.OK,
    *,
    head_only: bool = False,
) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    if not head_only:
        handler.wfile.write(body)


def parse_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    return json.loads(raw.decode("utf-8") or "{}")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def normalize_goal_cell(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return int(value[0]), int(value[1])
    except (TypeError, ValueError):
        return None


def normalize_goal_world(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def semantic_goal_label(index: int) -> str:
    labels = [
        "near_start",
        "inspection_point_a",
        "inspection_point_b",
        "corridor_left",
        "corridor_forward",
        "return_check",
    ]
    return labels[index] if index < len(labels) else f"inspection_point_{index + 1:02d}"


def episode_history_candidates(validation_report: Path, output_dir: Path) -> list[Path]:
    if validation_report.resolve() == DEFAULT_VALIDATION_REPORT.resolve():
        return EPISODE_HISTORY_CANDIDATES
    if not output_dir.is_dir():
        return []
    prefix = validation_report.stem
    return [
        path
        for path in sorted(output_dir.glob(f"{prefix}*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        if path.resolve() != validation_report.resolve()
    ][:8]


def load_semantic_goals(validation_report: Path, output_dir: Path) -> list[dict[str, Any]]:
    goals: list[dict[str, Any]] = []
    seen_cells: set[tuple[int, int]] = set()

    for report_path in episode_history_candidates(validation_report, output_dir):
        if not report_path.is_file():
            continue
        try:
            report = read_json(report_path)
        except Exception:
            continue
        for episode in report.get("episodes", []):
            cell = normalize_goal_cell(episode.get("goal_cell"))
            goal_xy = normalize_goal_world((episode.get("goal_xy") or [])[:2])
            if cell is None or goal_xy is None or cell in seen_cells:
                continue
            seen_cells.add(cell)
            goals.append(
                {
                    "label": semantic_goal_label(len(goals)),
                    "cell": [cell[0], cell[1]],
                    "world": {"x": goal_xy[0], "y": goal_xy[1]},
                    "source": report_path.name,
                    "lastResult": episode.get("result"),
                    "lastReason": episode.get("reason"),
                }
            )
            if len(goals) >= 8:
                return goals

    if validation_report.is_file():
        try:
            validation = read_json(validation_report)
            for goal in validation.get("map", {}).get("goals", []):
                goal_xy = normalize_goal_world(goal[:2])
                if goal_xy is None:
                    continue
                cell = None
                # The runner can resolve world XY, but WebUI benefits from a cell when possible.
                for existing in goals:
                    if abs(existing["world"]["x"] - goal_xy[0]) < 1e-4 and abs(existing["world"]["y"] - goal_xy[1]) < 1e-4:
                        cell = tuple(existing["cell"])
                        break
                if cell is not None and cell in seen_cells:
                    continue
                goals.append(
                    {
                        "label": semantic_goal_label(len(goals)),
                        "cell": [cell[0], cell[1]] if cell is not None else None,
                        "world": {"x": goal_xy[0], "y": goal_xy[1]},
                        "source": validation_report.name,
                        "lastResult": None,
                        "lastReason": None,
                    }
                )
        except Exception:
            pass

    return goals


class WorldNavController:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.process: subprocess.Popen[bytes] | None = None
        self.started_at_ms: int | None = None
        self.current_task: dict[str, Any] | None = None
        self.last_report_path: Path | None = None
        self.last_log_path: Path | None = None
        self.last_error: str | None = None
        self.semantic_goals = load_semantic_goals(Path(args.validation_report), Path(args.output_dir))

    def status(self) -> dict[str, Any]:
        running = self.process is not None and self.process.poll() is None
        return_code = None if self.process is None else self.process.poll()
        last_report = self._load_last_report()
        return {
            "mode": "world-nav-module",
            "ready": Path(self.args.validation_report).is_file(),
            "running": running,
            "returnCode": return_code,
            "currentTask": self.current_task,
            "startedAtMs": self.started_at_ms,
            "lastReportPath": str(self.last_report_path) if self.last_report_path else None,
            "lastLogPath": str(self.last_log_path) if self.last_log_path else None,
            "lastError": self.last_error,
            "semanticGoals": self.semantic_goals,
            "lastReport": last_report,
        }

    def _load_last_report(self) -> dict[str, Any] | None:
        if not self.last_report_path or not self.last_report_path.is_file():
            return None
        try:
            report = read_json(self.last_report_path)
        except Exception:
            return None
        return {
            "counts": report.get("counts"),
            "episodeCount": report.get("episode_count"),
            "goalSource": report.get("goal_source"),
            "episodes": report.get("episodes", [])[-3:],
        }

    def stop(self) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception as exc:
                self.last_error = str(exc)
            try:
                self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=4)
        return self.status()

    def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("A navigation task is already running. Stop it first.")

        command = str(payload.get("command") or "start_pointgoal")
        goal_label = str(payload.get("goalLabel") or payload.get("semanticGoal") or "")
        semantic_goal = str(payload.get("semanticGoal") or "") or None
        goal_cell = normalize_goal_cell(payload.get("goalCell"))
        goal_world = normalize_goal_world(payload.get("goalWorld"))

        if command == "start_semantic":
            semantic_goal = semantic_goal or goal_label or (self.semantic_goals[0]["label"] if self.semantic_goals else None)
            selected = next((goal for goal in self.semantic_goals if goal["label"] == semantic_goal), None)
            if selected is None:
                raise ValueError(f"Unknown semantic goal: {semantic_goal}")
            goal_label = selected["label"]
            goal_cell = normalize_goal_cell(selected.get("cell"))
            world = selected.get("world")
            goal_world = normalize_goal_world([world.get("x"), world.get("y")]) if isinstance(world, dict) else None
        elif command == "start_pointgoal":
            if goal_cell is None and goal_world is None:
                selected = self.semantic_goals[0] if self.semantic_goals else None
                if selected is None:
                    raise ValueError("No default point goal is available.")
                goal_label = str(selected["label"])
                goal_cell = normalize_goal_cell(selected.get("cell"))
                world = selected.get("world")
                goal_world = normalize_goal_world([world.get("x"), world.get("y")]) if isinstance(world, dict) else None
        else:
            raise ValueError(f"Unsupported command: {command}")

        if goal_cell is None and goal_world is None:
            raise ValueError("Goal must resolve to goalCell or goalWorld.")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path(self.args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"world_nav_module_{stamp}.json"
        log_path = output_dir / f"world_nav_module_{stamp}.log"
        goal_args = ""
        if goal_cell is not None:
            goal_args = f" --goal-cell={shell_quote(f'{goal_cell[0]},{goal_cell[1]}')}"
        elif goal_world is not None:
            goal_args = f" --goal-world={shell_quote(f'{goal_world[0]},{goal_world[1]}')}"

        label_args = ""
        if goal_label:
            label_args += f" --goal-label {shell_quote(goal_label)}"
        if semantic_goal:
            label_args += f" --semantic-goal {shell_quote(semantic_goal)}"

        runner = REPO_ROOT / "scripts" / "run_minimal_nav_episodes.py"
        viewer_args = ""
        if self.args.viewer:
            viewer_args = (
                f" --step-sleep {float(self.args.step_sleep)}"
                f" --episode-hold-sec {float(self.args.episode_hold_sec)}"
                f" --hold-open-sec {float(self.args.hold_open_sec)}"
            )
        shell_command = (
            f"source {shell_quote(str(self.args.isaac_env))}; "
            "export OMNI_KIT_ACCEPT_EULA=YES; "
            '"$ISAACSIM_PYTHON_EXE" '
            f"{shell_quote(str(runner))} "
            f"{'' if self.args.viewer else '--headless '} "
            f"--device {shell_quote(str(self.args.device))} "
            f"--robot-model {shell_quote(str(self.args.robot_model))} "
            f"--validation-report {shell_quote(str(self.args.validation_report))} "
            f"--output {shell_quote(str(report_path))} "
            "--episode-limit 1 "
            f"--timeout-sec {float(self.args.timeout_sec)} "
            f"--goal-tolerance {float(self.args.goal_tolerance)} "
            f"--max-speed {float(self.args.max_speed)} "
            f"--bridge-url {shell_quote(str(self.args.bridge_url))} "
            f"--bridge-publish-rate-hz {float(self.args.bridge_publish_rate_hz)}"
            f"{viewer_args}{goal_args}{label_args}"
        )

        log_handle = log_path.open("wb")
        self.process = subprocess.Popen(
            ["bash", "-lc", shell_command],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT),
            preexec_fn=os.setsid,
        )
        log_handle.close()

        self.started_at_ms = now_ms()
        self.last_report_path = report_path
        self.last_log_path = log_path
        self.last_error = None
        self.current_task = {
            "command": command,
            "goalLabel": goal_label or None,
            "semanticGoal": semantic_goal,
            "goalCell": [goal_cell[0], goal_cell[1]] if goal_cell is not None else None,
            "goalWorld": {"x": goal_world[0], "y": goal_world[1]} if goal_world is not None else None,
            "reportPath": str(report_path),
            "logPath": str(log_path),
        }
        return self.status()


class Handler(BaseHTTPRequestHandler):
    controller: WorldNavController

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_HEAD(self) -> None:
        self._handle_get(head_only=True)

    def do_GET(self) -> None:
        self._handle_get(head_only=False)

    def _handle_get(self, *, head_only: bool) -> None:
        if self.path == "/status":
            json_response(self, self.controller.status(), head_only=head_only)
            return
        if self.path == "/semantic_goals.json":
            json_response(self, {"goals": self.controller.semantic_goals}, head_only=head_only)
            return
        if self.path == "/last_report.json":
            json_response(self, {"lastReport": self.controller._load_last_report()}, head_only=head_only)
            return
        json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND, head_only=head_only)

    def do_POST(self) -> None:
        try:
            payload = parse_json_body(self)
            if self.path == "/control":
                command = str(payload.get("command") or "")
                if command == "stop":
                    json_response(self, self.controller.stop())
                    return
                json_response(self, self.controller.start(payload))
                return
            json_response(self, {"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, RuntimeError) as exc:
            self.controller.last_error = str(exc)
            json_response(self, {"error": str(exc), "status": self.controller.status()}, HTTPStatus.BAD_REQUEST)
        except json.JSONDecodeError as exc:
            json_response(self, {"error": f"Invalid JSON: {exc}"}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    args = parse_args()
    controller = WorldNavController(args)
    Handler.controller = controller
    server = ThreadingHTTPServer(("0.0.0.0", int(args.port)), Handler)
    try:
        print(f"world_nav_module_server listening on http://0.0.0.0:{args.port}", flush=True)
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        server.server_close()


if __name__ == "__main__":
    main()
