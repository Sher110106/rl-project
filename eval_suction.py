import time
import argparse
import cv2
import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from pick_place_env_suction import PickPlaceSuctionEnv

def evaluate(model_path, num_episodes=5, slow_down=0.05, save_video=True):
    # Initialize the environment
    env = PickPlaceSuctionEnv(render_mode="human")
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)
    
    print(f"Starting evaluation for {num_episodes} episodes...")
    
    # Initialize video writer
    video_writer = None
    if save_video:
        # Define the codec and create VideoWriter object
        # 'XVID' or 'MJPG' usually work well on Windows
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter('eval_video.avi', fourcc, 20.0, (640, 480))
        print("Recording video to eval_video.avi using OpenCV...")

    try:
        for i in range(num_episodes):
            obs, _ = env.reset()
            
            # obs[17:19] are the X and Y coordinates of the object
            start_x, start_y = obs[17], obs[18]
            print(f"Episode {i+1} | Cube starting at X: {start_x:.3f}, Y: {start_y:.3f}")
            
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                # Use deterministic=True for evaluation to get the best behavior
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
                # Capture frame if saving video
                if video_writer is not None:
                    # Get camera image from PyBullet
                    view_matrix = p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=[0.4, 0.0, 0.2],
                        distance=1.5,
                        roll=0,
                        pitch=-30,
                        yaw=45,
                        upAxisIndex=2
                    )
                    proj_matrix = p.computeProjectionMatrixFOV(
                        fov=60, aspect=640/480, nearVal=0.1, farVal=100.0
                    )
                    (_, _, px, _, _) = p.getCameraImage(
                        width=640, height=480,
                        viewMatrix=view_matrix,
                        projectionMatrix=proj_matrix,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL
                    )
                    rgb_array = np.array(px, dtype=np.uint8)
                    rgb_array = np.reshape(rgb_array, (480, 640, 4))
                    rgb_array = rgb_array[:, :, :3] # Remove alpha channel
                    # Convert RGB to BGR for OpenCV
                    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr_array)

                # Slow down the simulation
                time.sleep(slow_down) 
                
            print(f"Episode {i+1} Finished | Total Reward: {episode_reward:.2f} | Success: {info.get('cube_tray_dist', 1.0) < 0.05}")
    finally:
        if video_writer is not None:
            video_writer.release()
        env.close()
        print(f"Evaluation complete. Video saved as eval_video.avi")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="results/suction_gripper/seed0/models/best_model/best_model.zip", help="Path to the model zip file")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--slow", type=float, default=0.05, help="Seconds to sleep between steps")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    args = parser.parse_args()
    
    evaluate(args.model, args.episodes, args.slow, not args.no_video)
