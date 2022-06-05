# Author: Siddharth Palani Natarajan 

# ME 468 - Traffic Sign Detection for Autonomous Vehicle

#########################################################

import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='control_stack',
            namespace='',
            executable='simulation_final_project',
            name='simulation_final_project',
            on_exit=launch.actions.Shutdown(),
            parameters=[
                {"visualize":True},
                {"save":True},
                {"initial_location":[0.0,0.0,1.6]},
                {"initial_orientation":0.0},
                {"publish_state_directly":True},
                {"save_name":"dsine2"},
                {"max_duration": 30.0},
            ],
        ),
        Node(
            package='control_stack',
            namespace='',
            executable='control_path_follower_final_project',
            name='control_path_follower_final_project',
            parameters=[
                {"use_sim_time": True},
                {"visualize":False},
                {"vis_with_buffer":30.0},
                {"steering_kp": .2},
                {"steering_kd": 0.0},
                {"steering_ki": 0.0},
                {"lookahead": 4.77},
                {"speed_kp": 0.2},
                {"speed_kd": 0.0},
                {"speed_ki": 0.0},
                {"target_speed": 10.0}
            ],
        ),
        Node(
            package='control_stack',
            namespace='',
            executable='path_planner_from_file',
            name='path_planner_from_file',
            parameters=[
                {"use_sim_time": True},
                {"path_file": "data/path_planner/sin_curve_2.csv"}
            ]
        )
    ])


