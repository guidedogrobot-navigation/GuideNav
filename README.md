# GuideNav 🦮
A visual teach-and-repeat system using topological mapping for autonomous robot navigation (RGB-only, untethered).

## 🚀 Deployment

### 🔧 Hardware Requirements
- **Jetson AGX Orin** - Main compute unit
- **Laptop** - Data saving and code execution

### 🏃 Quick Start
[Jetson Orin (ssh)]
```bash
# Mount SSD
mt2

# Run navigation system (opens tmux w/ (1) camera streaming, (2) action command, (3) navigation script)
navigate

# Detach from session
Ctrl+b, d
```
[Laptop]
```bash
# Connect to local WiFi
outnavnet - daros123

# Subscribe ROS2 Msgs for data saving (open terminal and make sure you have topomap in data saving PC)
python3 save_data.py --topo_dir ~/Downloads/sampled_topo
python save_data.py --topo_dir ~/Downloads/topo_gail2/topo_gail2parking_20250707_003014_3m_mine2
```

## Data Collection
[Jetson Orin (ssh)]
```bash
# Mount SSD
mt2

# Setup Go2 Ros2
tmuxx
cd ~/go2_ws/install && source setup.bash
ros2 launch go2_robot_sdk robot.launch.py

# Setup camera
doc
cam

# save w/ joint pos
ros2 bag record -o raw_bag_$(date +%Y%m%d_%H%M%S)   /camera/color/image_raw   /camera/color/camera_info   /camera/aligned_depth_to_color/image_raw   /camera/aligned_depth_to_color/camera_info   /visual_slam/tracking/odometry  /tf   /tf_static /lowstate

ros2 bag record -o raw_bag_$(date +%Y%m%d_%H%M%S)   /d435i/color/image_raw   /d435i/color/camera_info   /d435i/aligned_depth_to_color/image_raw   /d435i/aligned_depth_to_color/camera_info   /visual_slam/tracking/odometry  /tf   /tf_static /lowstate

ros2 bag record -o tactile_$(date +%Y%m%d_%H%M%S)   /d435/color/image_raw   /d435/color/camera_info   /d435i/aligned_depth_to_color/image_raw  /d435i/color/image_raw   /d435i/color/camera_info  /d435i/aligned_depth_to_color/camera_info   /d435i/imu   /visual_slam/tracking/odometry   /tf   /tf_static

# Detach from session
Ctrl+b, d
(mac) tmux -> new pane (ctrl+a+v) -> tmux detach-client

# Tactile paving bag extraction
orin2@orin2:~/Repository/GuideNav/src/sensor$ python batch_extract.py /media/2t/ijrr/bags_tactile_day1 extract_data_two.py

# Tactile paving model TensorRT w/ quantization
yolo export model=.pt format=engine int8=True
```



## Data Visualization
### Foxglove
[Jetson Orin (ssh)]
```bash
# Play ros2 bag and launch foxglove (need to source go2_ws/install.setup.bash for /lowstate topic)
ros2 bag play raw_bag_XXXX
ros2 launch foxglove_bridge foxglove_bridge_launch.xml
```
[Desktop]
```bash
# Foxglove UI (10.42.0.1)
```

### Data Extract
```bash
cd ~/Repository/GuideNav/src/sensor
python extract_data.py
ros2 bag play raw_XXX
```

### Extracted Data Visualization
```bash
cd /home/orin2/Repository/GuideNav/src/sensor
python plot_dist_traj.py /media/2t/ijrr/20250701_095813/odom.csv --output northsquare_long.png --no-show
```

## Topomap construction (choose distance & yaw; default is 0.5m and 15 deg)
```bash
cd /home/orin2/Repository/GuideNav/src/sensor
sudo python build_topomap.py /media/2t/ijrr/20250701_052514 /media/2t/ijrr/topo_20250701_052514 --distance 1.0
```


## 📊 Features

- **🎯 RGB-only Navigation** - No additional sensors required
- **🗺️ Topological Mapping** - Efficient map representation
- **🔋 Untethered Operation** - Full autonomous deployment
- **⚡ Real-time Processing** - Optimized for Jetson platform (perception)



## 📁 (TODO) Clean Project Structure
```
GuideNav/
├── src/
│   ├── navigation/     # Core VTR algorithm
│   ├── mapping/        # Topological map building
│   └── vision/         # Perception methods
└── data/              # Navigation data
```
