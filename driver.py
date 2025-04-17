import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

# Function that duplicates a value 'what' for 'times' times
def dup(what, times): 
  return [what for _ in range(0, times)]

# Utility functions
def sqrMag(vec): 
  return vec.dot(vec)

def mag(vec): 
  return np.sqrt(sqrMag(vec))

# ==== Plot Results ====
def plotResults(t, dt, targetPosition, robotPosition, robotTheta, robotVelocity, targetTheta, relativeOrientation):
  timeAxis = np.array(t) * dt  # convert time steps to seconds

  # Create a figure with a 2x2 grid of subplots
  fig, axs = plt.subplots(2, 2, figsize=(12, 10))

  # --- 1. Trajectories of Target and Robot ---
  axs[0, 0].plot(targetPosition[:, 0], targetPosition[:, 1], label="Target Trajectory", color='blue')
  axs[0, 0].plot(robotPosition[:, 0], robotPosition[:, 1], label="Robot Trajectory", color='orange')
  axs[0, 0].set_xlabel("X Position")
  axs[0, 0].set_ylabel("Y Position")
  axs[0, 0].set_title("Trajectory Comparison")
  axs[0, 0].legend()
  axs[0, 0].grid(True)
  axs[0, 0].axis("equal")

  # --- 2. Tracking Error ---
  trackingError = np.linalg.norm(targetPosition - robotPosition, axis=1)
  axs[0, 1].plot(timeAxis, trackingError, label="Tracking Error", color='red')
  axs[0, 1].set_xlabel("Time (s)")
  axs[0, 1].set_ylabel("Distance")
  axs[0, 1].set_title("Tracking Error Over Time")
  axs[0, 1].grid(True)

  # --- 3. Robot Orientation, Target Orientation, and Relative Orientation Over Time ---

  axs[1, 0].plot(timeAxis, robotTheta, label="Robot Orientation", color='green')
  axs[1, 0].plot(timeAxis, targetTheta, label="Target Orientation", color='blue')
  axs[1, 0].plot(timeAxis, relativeOrientation, label="Relative Orientation", color='purple')
  axs[1, 0].set_xlabel("Time (s)")
  axs[1, 0].set_ylabel("Angle (rad)")
  axs[1, 0].set_title("Orientation Over Time")
  axs[1, 0].legend()
  axs[1, 0].grid(True)

  # --- 4. Robot Velocity Over Time ---
  axs[1, 1].plot(timeAxis, robotVelocity, label="Robot Velocity", color='purple')
  axs[1, 1].set_xlabel("Time (s)")
  axs[1, 1].set_ylabel("Velocity (units/s)")
  axs[1, 1].set_title("Robot Velocity Over Time")
  axs[1, 1].grid(True)

  # Adjust layout to prevent overlap
  plt.tight_layout()

  # Show all plots in the same window
  plt.show()


# ======== MAIN PROGRAM =========
if __name__ == "__main__":
  
  # Change Parameter for different target types:
  typeNum = 2
  types = ["circle", "linear", "sin", "circleN", "linearN", "sinN"]
  targetType = types[typeNum-1]


  # Define all parameters inside main
  n = 2  # Number of dimensions
  dt = 0.05  # Time step
  t = range(0, 10 * 100, int(dt * 100))  # Simulation steps
  lambdaGain = 8.5  # Attractive potential gain
  maxRobotVelocity = 50  # Maximum robot velocity
  error = np.array(dup(0, len(t)), np.float64)

  # ========== Set Virtual Target ==========
  targetPosition = np.array(dup([0, 0], len(t)), np.float64)  # Initial target positions
  targetVelocity = 1.2  # Constant target velocity
  targetTheta = np.array(dup(0, len(t)), np.float64)  # Target heading
  targetDiff = np.array(dup([0, 0], len(t)), np.float64)

  # ========== Set Robot ==========
  robotPosition = np.array(dup([0, 0], len(t)), np.float64)  # Initial robot position
  robotVelocity = np.array(dup(0, len(t)), np.float64)  # Robot velocity
  robotTheta = np.array(dup(0, len(t)), np.float64)  # Robot heading

  # ========== Relative States ==========
  relativePosition = np.array(dup([0, 0], len(t)), np.float64)  # Relative position
  relativeVelocity = np.array(dup([0, 0], len(t)), np.float64)  # Relative velocity
  relativeOrientation = np.array(dup(0, len(t)), np.float64)  # Robot heading

  # ==== Compute Initial Relative States ====
  relativePosition[0, :] = targetPosition[0, :] - robotPosition[0, :]
  relativeVelocity[0, :] = [
    targetVelocity * np.cos(targetTheta[0]) - robotVelocity[0] * np.cos(robotTheta[0]),
    targetVelocity * np.sin(targetTheta[0]) - robotVelocity[0] * np.sin(robotTheta[0])
  ]

  # ==== Noise Parameters ====
  noiseMean = 0.5
  noiseStd = 0.5  # Try 0.2 as well

  # ==== Variables for Calculation ====
  phi = np.array(dup(0, len(t)), np.float64)

  modRelative = False

  # Simulation loop
  for i in range(0, len(t)):
    time = (t[i] * 1.495) / 100

    if targetType == "circle":
      # ++++++++ CIRCULAR TRAJECTORY WITHOUT NOISE +++++++++
      targetVelocity = 1.2
      targetPosX = 60 - 15 * np.cos(time)
      targetPosY = 30 + 15 * np.sin(time)
      targetPosition[i, :] = [targetPosX, targetPosY]

    elif targetType == "circleN":
      # CIRCULAR TRAJECTORY WITH NOISE (UNCOMMENT TO USE)
      targetVelocity = 1.2
      targetPosX = 60 - 15 * np.cos(time) + random.uniform(-noiseStd, noiseStd) + noiseMean
      targetPosY = 30 + 15 * np.sin(time) + random.uniform(-noiseStd, noiseStd) + noiseMean
      targetPosition[i, :] = [targetPosX, targetPosY]

    elif targetType == "linear":
      # Linear trajectory (constant velocity up-left)
      targetVelocity = 10
      vx = targetVelocity / np.sqrt(2)
      vy = targetVelocity / np.sqrt(2)
      targetPosX = 150 - vx * time  
      targetPosY = 0 + vy * time
      targetPosition[i, :] = [targetPosX, targetPosY]

    elif targetType == "linearN":
      targetVelocity = 15
      # Set base heading (45 degrees for diagonal movement)
      baseAngle = np.pi / 4 * 3 # 45 degrees in radians

      # Add small angular noise to the heading
      maxAngleNoise = np.radians(30)  # max deviation of ±30 degrees
      angleNoise = np.clip(np.random.normal(0, noiseStd), -maxAngleNoise, maxAngleNoise)
      noisyAngle = baseAngle + angleNoise

      # Compute velocity components with noise
      vx = targetVelocity * np.cos(noisyAngle)
      vy = targetVelocity * np.sin(noisyAngle)

      # Initialize starting position (150, 0)
      if i == 0:  # First iteration
          targetPosX = 150
          targetPosY = 0
      else:
          # Update position based on previous position
          targetPosX = targetPosition[i-1, 0] + vx * dt
          targetPosY = targetPosition[i-1, 1] + vy * dt

      # Assign to target position
      targetPosition[i, :] = [targetPosX, targetPosY]
      modRelative = True

    elif targetType == "sin":
      # Sine wave trajectory
      targetPosX = 30 + targetVelocity * time * 2
      targetPosY = 30 + 20 * np.sin(1 * time)
      targetPosition[i, :] = [targetPosX, targetPosY]

    elif targetType == "sinN":
      # Sine wave trajectory
      targetPosX = 30 + targetVelocity * time * 2 + random.uniform(-noiseStd, noiseStd) + noiseMean
      targetPosY = 30 + 20 * np.sin(1 * time) + random.uniform(-noiseStd, noiseStd) + noiseMean
      targetPosition[i, :] = [targetPosX, targetPosY]
      modRelative = True

    else:
      print("Type not recognized")
      break

    print([targetPosX, targetPosY])

    # .. (Add your robot control logic here)
    # ====== Compute control (potential field approach) ======
    relativePosition[i, :] = targetPosition[i, :] - robotPosition[i - 1, :]

    # Desired velocity from potential field
    targetVelVec = targetVelocity * np.array([
      np.cos(targetTheta[i]),
      np.sin(targetTheta[i])
    ])
    desiredVelocity = lambdaGain * relativePosition[i, :] + targetVelVec

    # Clip desired velocity if it exceeds max
    if mag(desiredVelocity) > maxRobotVelocity:
      desiredVelocity = (desiredVelocity / mag(desiredVelocity)) * maxRobotVelocity

    # Update robot position and velocity
    robotVelocity[i] = mag(desiredVelocity)
    robotPosition[i, :] = robotPosition[i - 1, :] + desiredVelocity * dt

    # Update robot heading
    robotTheta[i] = np.arctan2(desiredVelocity[1], desiredVelocity[0])

    # Update target heading
    targetDiff[i, :] = targetPosition[i, :] - targetPosition[i - 1, :]
    targetTheta[i] = np.arctan2(targetDiff[i, 1], targetDiff[i, 0])

    # Compute and store relative heading (angle of relative velocity vector)
    xrv = targetVelocity * np.cos(targetTheta[0]) - robotVelocity[0] * np.cos(robotTheta[0])
    yrv = targetVelocity * np.sin(targetTheta[i]) - robotVelocity[i] * np.sin(robotTheta[i])
    relativeOrientation[i] = np.arctan2(yrv, xrv) 

    if modRelative:
      relativeOrientation[i] = relativeOrientation[i] % (2 * np.pi)

  # After the simulation loop, call the plot function to visualize the results
  plotResults(t, dt, targetPosition, robotPosition, robotTheta, robotVelocity, targetTheta, relativeOrientation)
