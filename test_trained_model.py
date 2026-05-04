#!/usr/bin/env python3
"""
Evaluate and showcase the best trained SAC model in this repo.

Behavior:
- Finds the best saved checkpoint by scanning results/suction_gripper/seed*/eval/evaluations.npz
- Runs deterministic evaluation episodes
- If a GUI display is available and --live is requested, shows PyBullet live
- Otherwise records an MP4 of the first successful episode, or the best-return episode
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pybullet as p
from stable_baselines3 import SAC

from pick_place_env_suction import PickPlaceSuctionEnv


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results" / "suction_gripper"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "test_runs"


@dataclass
class ModelSelection:
    seed_name: str
    model_path: Path
    eval_path: Path
    peak_mean_reward: float
    final_mean_reward: float
    best_eval_index: int


@dataclass
class EpisodeResult:
    episode_index: int
    total_reward: float
    steps: int
    success: bool
    frames: list[np.ndarray]


def find_best_model(results_root: Path) -> ModelSelection:
    candidates: list[ModelSelection] = []

    for seed_dir in sorted(results_root.glob("seed*")):
        eval_path = seed_dir / "eval" / "evaluations.npz"
        model_path = seed_dir / "models" / "best_model" / "best_model.zip"
        if not eval_path.exists() or not model_path.exists():
            continue

        data = np.load(eval_path)
        mean_rewards = data["results"].mean(axis=1)
        candidates.append(
            ModelSelection(
                seed_name=seed_dir.name,
                model_path=model_path,
                eval_path=eval_path,
                peak_mean_reward=float(mean_rewards.max()),
                final_mean_reward=float(mean_rewards[-1]),
                best_eval_index=int(mean_rewards.argmax()),
            )
        )

    if not candidates:
        raise FileNotFoundError(
            "No trained suction-gripper checkpoints were found under results/suction_gripper/."
        )

    return max(candidates, key=lambda item: item.peak_mean_reward)


def capture_frame(env: PickPlaceSuctionEnv, width: int, height: int) -> np.ndarray:
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.5, 0.0, 0.12],
        distance=1.15,
        yaw=45,
        pitch=-35,
        roll=0,
        upAxisIndex=2,
    )
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=float(width) / float(height),
        nearVal=0.1,
        farVal=3.1,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=env._client,
    )
    frame = np.asarray(rgba, dtype=np.uint8)[:, :, :3]
    return frame


def run_episode(
    model: SAC,
    episode_index: int,
    live: bool,
    capture_video: bool,
    fps: int,
    width: int,
    height: int,
    max_steps: int,
) -> EpisodeResult:
    env = PickPlaceSuctionEnv(render_mode="human" if live else None)
    frames: list[np.ndarray] = []

    try:
        obs, _ = env.reset()
        total_reward = 0.0
        success = False

        if capture_video:
            frames.append(capture_frame(env, width, height))

        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)

            if capture_video:
                frames.append(capture_frame(env, width, height))

            if live:
                time.sleep(1.0 / float(fps))

            if terminated or truncated:
                success = bool(terminated and info.get("cube_tray_dist", 999.0) < 0.05)
                return EpisodeResult(
                    episode_index=episode_index,
                    total_reward=total_reward,
                    steps=step,
                    success=success,
                    frames=frames,
                )

        return EpisodeResult(
            episode_index=episode_index,
            total_reward=total_reward,
            steps=max_steps,
            success=success,
            frames=frames,
        )
    finally:
        env.close()


def save_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264") as writer:
        for frame in frames:
            writer.append_data(frame)


def write_summary(
    output_dir: Path,
    selection: ModelSelection,
    chosen_episode: EpisodeResult,
    all_episodes: list[EpisodeResult],
    video_path: Path | None,
) -> Path:
    summary = {
        "selected_seed": selection.seed_name,
        "model_path": str(selection.model_path.relative_to(REPO_ROOT)),
        "eval_path": str(selection.eval_path.relative_to(REPO_ROOT)),
        "peak_mean_reward": selection.peak_mean_reward,
        "final_mean_reward": selection.final_mean_reward,
        "best_eval_index": selection.best_eval_index,
        "chosen_episode": {
            "episode_index": chosen_episode.episode_index,
            "total_reward": chosen_episode.total_reward,
            "steps": chosen_episode.steps,
            "success": chosen_episode.success,
        },
        "episodes": [
            {
                "episode_index": episode.episode_index,
                "total_reward": episode.total_reward,
                "steps": episode.steps,
                "success": episode.success,
            }
            for episode in all_episodes
        ],
        "video_path": str(video_path.relative_to(REPO_ROOT)) if video_path else None,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live PyBullet GUI playback. Requires a working DISPLAY inside the container.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode and record an MP4 instead of attempting live playback.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.live and args.headless:
        print("Choose either --live or --headless, not both.", file=sys.stderr)
        return 2

    display_available = bool(os.environ.get("DISPLAY"))
    live = args.live or (display_available and not args.headless)
    capture_video = not live

    selection = find_best_model(RESULTS_ROOT)
    print(
        f"Selected best trained model: {selection.seed_name} "
        f"(peak eval mean reward = {selection.peak_mean_reward:.3f})"
    )
    print(f"Checkpoint: {selection.model_path.relative_to(REPO_ROOT)}")

    model = SAC.load(str(selection.model_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    episodes: list[EpisodeResult] = []
    successful_episode: EpisodeResult | None = None
    best_episode: EpisodeResult | None = None

    for episode_index in range(args.episodes):
        episode = run_episode(
            model=model,
            episode_index=episode_index,
            live=live,
            capture_video=capture_video,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_steps=args.max_steps,
        )
        episodes.append(
            EpisodeResult(
                episode_index=episode.episode_index,
                total_reward=episode.total_reward,
                steps=episode.steps,
                success=episode.success,
                frames=[],
            )
        )
        if best_episode is None or episode.total_reward > best_episode.total_reward:
            best_episode = episode

        status = "SUCCESS" if episode.success else "partial"
        print(
            f"Episode {episode_index + 1}/{args.episodes}: "
            f"reward={episode.total_reward:.3f}, steps={episode.steps}, {status}"
        )

        if episode.success:
            successful_episode = episode
            break

    chosen_episode = successful_episode or best_episode
    assert chosen_episode is not None

    video_path: Path | None = None
    if capture_video:
        video_name = "successful_rollout.mp4" if chosen_episode.success else "best_rollout.mp4"
        video_path = args.output_dir / video_name
        save_video(chosen_episode.frames, video_path, args.fps)
        print(f"Saved video: {video_path.relative_to(REPO_ROOT)}")
    else:
        print("Live playback completed in the PyBullet GUI window.")

    summary_path = write_summary(
        output_dir=args.output_dir,
        selection=selection,
        chosen_episode=chosen_episode,
        all_episodes=episodes,
        video_path=video_path,
    )
    print(f"Saved summary: {summary_path.relative_to(REPO_ROOT)}")

    if not display_available and live:
        print("DISPLAY was not available, so live playback may fail inside a bare container.")

    if chosen_episode.success:
        print("The showcased rollout achieved a successful placement.")
    else:
        print("No full success occurred in this test batch, so the best-reward rollout was saved.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
