

# wall_follower_pkg


Ros2 package for MIT RACECAR wall following. Subscribes to `/scan` for LaserScanStamped messages and publishes AckermannDriveStamped messages to `/vesc/high_level/input/nav_0`.

Also publishes debug information to `/wall_marker`, `/fit_marker`, `/wall_distance`, and `/wall_distance_marker`.

Uses RANSAC for wall detection and a PD algorithm for driving.


Note: the package name is `wall_follower`.

Build:

```
cd ~/racecar_ws && colcon build --packages-select wall_follower && source install/setup.bash
```

Run:
```
ros2 launch wall_follower launch_test_sim.launch.py
```


