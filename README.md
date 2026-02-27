# ArduPilot + Gazebo Classic + ROS 2 Setup Guide
This repository provides a step-by-step walkthrough to set up an autonomous drone simulation environment using ArduPilot (SITL), Gazebo Classic, and ROS 2 (MAVROS).

#  Prerequisites
OS: Ubuntu 20.04 or 22.04

Middleware: ROS 2 (Foxy, Humble, or Jazzy)

Simulator: Gazebo Classic (Gazebo 11)

##  Step 1: Install ArduPilot and SITL
First, you need to install the ArduPilot firmware and its simulation dependencies.

1. **Clone the ArduPilot repository:**
   ```
   git clone [https://github.com/ArduPilot/ardupilot](https://github.com/ArduPilot/ardupilot)
   cd ardupilot
   git submodule update --init --recursive 
   ```
## 2. Install Dependencies:

   ```
./Tools/environment_install/install-prereqs-ubuntu.sh -y
. ~/.profile
 ```

## Step 2: Setup Gazebo Simulation (IQ Sim)
We use the iq_sim package for the world files and drone models.

Clone the IQ Simulation repo:
   ```
git clone https://github.com/Intelligent-Quads/iq_sim.git ~/iq_sim 

 ```

## Add models to Gazebo path:
Open your .bashrc: nano ~/.bashrc and add the following line at the end:
 ```

export GAZEBO_MODEL_PATH=$HOME/iq_sim/models:$GAZEBO_MODEL_PATH
source ~/.bashrc

 ```

## Step 3: Launching the Simulation
You will need three separate terminal tabs to run the full stack.

Terminal 1: Gazebo World
Launch the physics engine with the runway world:
 ```
gazebo --verbose $HOME/iq_sim/worlds/runway.world
 ```

## Terminal 2: ArduPilot SITL
Navigate to the ArduCopter directory and launch the flight controller. This links ArduPilot to the Gazebo Iris model.
 ```
cd ~/ardupilot/ArduCopter
python3 ../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --console --map

 ```

## Terminal 3: MAVROS (The ROS 2 Bridge)
Once the SITL is running and says "GPS Lock," launch MAVROS to allow ROS 2 to communicate with the drone.
```
ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://127.0.0.1:14550@

 ```
