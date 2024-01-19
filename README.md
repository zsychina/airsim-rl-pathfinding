# airsim-rl-pathfinding

project is still undergoing

## current bugs

- poor training performance
- reward seems in a steady decline

## state format

example

```txt
state: <MultirotorState>

{
    "collision":"<CollisionInfo>"{
        "has_collided":false,
        "impact_point":<Vector3r>{
            "x_val":0.0,
            "y_val":0.0,
            "z_val":0.0
        },
        "normal":<Vector3r>{
            "x_val":0.0,
            "y_val":0.0,
            "z_val":0.0
        },
        "object_id":-1,
        "object_name":"",
        "penetration_depth":0.0,
        "position":<Vector3r>{
            "x_val":0.0,
            "y_val":0.0,
            "z_val":0.0
        },
        "time_stamp":0
    },
    "gps_location":"<GeoPoint>"{
        "altitude":177.06935119628906,
        "latitude":47.6417694733832,
        "longitude":-122.13984290488607
    },
    "kinematics_estimated":"<KinematicsState>"{
        "angular_acceleration":<Vector3r>{
            "x_val":0.022923434153199196,
            "y_val":0.11459309607744217,
            "z_val":0.0001473926822654903
        },
        "angular_velocity":<Vector3r>{
            "x_val":-0.016605805605649948,
            "y_val":-0.0807669460773468,
            "z_val":-6.932737596798688e-05
        },
        "linear_acceleration":<Vector3r>{
            "x_val":0.5586466193199158,
            "y_val":-0.048002228140830994,
            "z_val":0.05122089385986328
        },
        "linear_velocity":<Vector3r>{
            "x_val":-0.38533058762550354,
            "y_val":0.0335688441991806,
            "z_val":0.001900549279525876
        },
        "orientation":"<Quaternionr>"{
            "w_val":-0.0595640167593956,
            "x_val":-0.028229856863617897,
            "y_val":0.004142099525779486,
            "z_val":0.9978166222572327
        },
        "position":<Vector3r>{
            "x_val":33.51864242553711,
            "y_val":24.203163146972656,
            "z_val":-55.06911849975586
        }
    },
    "landed_state":1,
    "rc_data":"<RCData>"{
        "is_initialized":false,
        "is_valid":false,
        "left_z":0.0,
        "pitch":0.0,
        "right_z":0.0,
        "roll":0.0,
        "switches":0,
        "throttle":0.0,
        "timestamp":0,
        "vendor_id":"",
        "yaw":0.0
    },
    "timestamp":1704569007234179840
}

```

## distance sensor format

```txt
distance sensor: <DistanceSensorData> {   'distance': 23.301977157592773,
    'max_distance': 40.0,
    'min_distance': 0.20000000298023224,
    'relative_pose': <Pose> {   'orientation': <Quaternionr> {   'w_val': 1.0,
    'x_val': 0.0,
    'y_val': 0.0,
    'z_val': 0.0},
    'position': <Vector3r> {   'x_val': 0.0,
    'y_val': 0.0,
    'z_val': -0.10000000149011612}},
    'time_stamp': 1704967869749285376}
```
