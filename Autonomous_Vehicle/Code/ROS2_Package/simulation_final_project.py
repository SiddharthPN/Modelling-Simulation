# Author: Siddharth Palani Natarajan 

# ME 468 - Traffic Sign Detection for Autonomous Vehicle

#########################################################

# =============================================================================
# Copyright (c) 2022, University of Wisconsin - Madison
# Copyright (c) 2022, Simulation-Based Engineering Lab
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
#
# =============================================================================
from pychrono.core import CH_C_PI_2
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from custom_msgs.msg import VehicleState, VehicleInput, VehicleSpeed
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
# from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, NavSatFix, Imu, MagneticField, PointCloud2
from builtin_interfaces.msg import Time

from ament_index_python.packages import get_package_share_directory

import pychrono as chrono
import pychrono.sensor as sens
import pychrono.vehicle as veh

import os, time
import numpy as np

# import pickle  # Required for Traffic Sign Detection Code to function
# import cv2     # Required for Computer Vision


class SimulationNode(Node):
    def __init__(self,term_condition):
        super().__init__('simulation_node')

        self.future = term_condition

        # Simulation node parameters (launch file if specified)
        self.declare_parameter('step_size', 1e-3)
        self.step_size = self.get_parameter(
            'step_size').get_parameter_value().double_value

        self.declare_parameter('terrainLength', 1000.0)
        self.terrainLength = self.get_parameter(
            'terrainLength').get_parameter_value().double_value

        self.declare_parameter('terrainWidth', 400.0)
        self.terrainWidth = self.get_parameter(
            'terrainWidth').get_parameter_value().double_value

        self.declare_parameter('save_training_data', False)
        self.save_training_data = self.get_parameter(
            'save_training_data').get_parameter_value().bool_value

        self.declare_parameter('visualize', False)
        self.visualize = self.get_parameter(
            'visualize').get_parameter_value().bool_value

        self.declare_parameter('save', False)
        self.save = self.get_parameter(
            'save').get_parameter_value().bool_value

        self.declare_parameter('sensors', False)
        self.sensors = self.get_parameter(
            'sensors').get_parameter_value().bool_value

        self.declare_parameter('publish_state_directly', False)
        self.publish_state_directly = self.get_parameter(
            'publish_state_directly').get_parameter_value().bool_value

        self.declare_parameter('initial_location', [0.0, 0.0, 1.6])
        self.initial_location = self.get_parameter(
            'initial_location').get_parameter_value().double_array_value

        self.declare_parameter('initial_orientation', 0.0)
        self.initial_orientation = self.get_parameter(
            'initial_orientation').get_parameter_value().double_value

        self.declare_parameter('object_location_file', "")
        self.object_location_file = self.get_parameter(
            'object_location_file').get_parameter_value().string_value

        self.declare_parameter('random_object_count', 0)
        self.random_object_count = self.get_parameter(
            'random_object_count').get_parameter_value().integer_value

        self.declare_parameter('obstacle_scale', [2.0,4.0])
        self.obstacle_scale = self.get_parameter(
            'obstacle_scale').get_parameter_value().double_array_value
        
        self.declare_parameter('max_duration', 120.0)
        self.max_duration = self.get_parameter(
            'max_duration').get_parameter_value().double_value

        self.declare_parameter('target_location', [0.0,0.0,100.0])
        self.target_location = self.get_parameter(
            'target_location').get_parameter_value().double_array_value
        self.target_location = chrono.ChVectorD(self.target_location[0],self.target_location[1],self.target_location[2])

        self.declare_parameter('save_name', "simulation_output")
        self.save_name = self.get_parameter(
            'save_name').get_parameter_value().string_value

        home_dir = os.getenv('HOME')
        save_dir = "me468_output"
        if(not os.path.exists(os.path.join(home_dir,save_dir))):
            os.mkdir(os.path.join(home_dir,save_dir))

        self.output_dir = os.path.join(home_dir,save_dir,self.save_name)
        if(not os.path.exists(self.output_dir)):
            os.mkdir(self.output_dir)

        # Read in share directory location
        self.package_share_directory = get_package_share_directory(
            'control_stack')

        # tracked class variables
        self.sim_step = 0
        self.vehicle = None
        self.terrain = None
        self.driver = None
        self.manager = None
        self.camera = None
        self.magnetometer = None
        self.gps = None
        self.rock_assets = []
        self.throttle = 0.0
        self.braking = 0.0
        self.steering = 0.0
        self.braking_time = 0.5
        self.throttle_time = 1.0
        self.steering_time = 1.0

        # publish frequencies
        self.clock_frequency = 100.0
        self.camera_frequency = 10.0
        self.mag_frequency = 50.0
        self.gps_frequency = 10.0
        self.state_frequency = 10.0
        self.lidar_frequency = 10.0

        self.dur1 = 0.0
        self.dur2 = 0.0

        self.clock_interval = np.round(
            1 / self.step_size / self.clock_frequency)
        self.camera_interval = np.round(
            1 / self.step_size / self.camera_frequency)
        self.mag_interval = np.round(1 / self.step_size / self.mag_frequency)
        self.gps_interval = np.round(1 / self.step_size / self.gps_frequency)
        self.lidar_interval = np.round(1 / self.step_size / self.lidar_frequency)
        self.state_interval = np.round(
            1 / self.step_size / self.state_frequency)

        self.save_interaval = np.round(
            1 / self.step_size / 100.0)

        # set seed for consistent results
        chrono.ChSetRandomSeed(1)

        # subscribers
        self.sub_cmds = self.create_subscription(
            VehicleInput, 'vehicle_cmd', self.input_callback, 10)

        # publishers
        self.pub_clock = self.create_publisher(Clock, 'clock', 10)
        #self.pub_vehiclespeed = self.create_publisher(VehicleSpeed, 'vehiclespeed', 10)
        if(self.sensors):
            # self.pub_image = self.create_publisher(Image, 'image', 10)
            self.pub_gps = self.create_publisher(NavSatFix, 'gps', 10)
            self.pub_mag = self.create_publisher(MagneticField, 'magnetic', 10)
            self.pub_lidar = self.create_publisher(PointCloud2, 'lidar', 10)
            self.pub_vehiclespeed = self.create_publisher(VehicleSpeed, 'vehiclespeed', 10) # Publisher Node to publish the detected speed via the Camera Sensor

        self.pub_state = self.create_publisher(VehicleState, 'state', 10)
        

        # self.bridge = CvBridge()

        # timers
        self.timer = self.create_timer(self.step_size, self.step_simulation)

        self.output_data = []

        # initialize the vehicle simulation
        self.initialize_simulation()


    def initialize_simulation(self):
        # set chrono data directories
        CHRONO_DATA_DIR = os.environ.get('CHRONO_DATA_DIR')
        chrono.SetChronoDataPath(CHRONO_DATA_DIR)
        veh.SetDataPath(os.path.join(CHRONO_DATA_DIR, "vehicle/"))

        # create the chrono vehicle (Nissan Patrol SUV)
        initLoc = chrono.ChVectorD(
            self.initial_location[0], self.initial_location[1], self.initial_location[2])
        initRot = chrono.Q_from_AngZ(self.initial_orientation)
        vehicle_file = veh.GetDataFile("Nissan_Patrol/json/suv_Vehicle.json")
        powertrain_file = veh.GetDataFile(
            'Nissan_Patrol/json/suv_ShaftsPowertrain.json')
        tire_file = veh.GetDataFile('Nissan_Patrol/json/suv_TMeasyTire.json')
        terrain_file = veh.GetDataFile('terrain/RigidPlane.json')

        # create chrono vehicle model
        self.vehicle = veh.WheeledVehicle(
            vehicle_file, chrono.ChContactMethod_SMC)
        self.vehicle.Initialize(chrono.ChCoordsysD(initLoc, initRot))
        self.vehicle.SetChassisVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetChassisCollide(False)
        self.vehicle.SetSuspensionVisualizationType(
            veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSteeringVisualizationType(
            veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)

        # Create the ground
        self.terrain = veh.RigidTerrain(self.vehicle.GetSystem())
        patch_mat = chrono.ChMaterialSurfaceSMC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)

        patch = self.terrain.AddPatch(patch_mat,
                                      chrono.ChVectorD(
                                          0, 0, 0), chrono.ChVectorD(0, 0, 1),
                                      self.terrainLength, self.terrainWidth)
        self.terrain.Initialize()
        visual_asset = chrono.CastToChVisualization(
            patch.GetGroundBody().GetAssets()[0])
        vis_mat = chrono.ChVisualMaterial()
        vis_mat.SetKdTexture(veh.GetDataFile("terrain/textures/grass.jpg"))
        vis_mat.SetSpecularColor(chrono.ChVectorF(.0, .0, .0))
        vis_mat.SetRoughness(1.0)
        vis_mat.SetUseSpecularWorkflow(False)
        visual_asset.material_list.push_back(vis_mat)
        
    # # Inserting Traffic Sign 
        
    def AddFixedObstacles(system) :
         # Create contact material, of appropriate type. Use default properties
         matNSC = chrono.ChMaterialSurfaceNSC()
         matSMC = chrono.ChMaterialSurfaceSMC()
         material = matNSC
         material2 = matSMC
            
    #     # Asphalt
         aspha = chrono.ChBodyEasyBox(10000,2,1,0.1,True,False,material2)
         aspha.SetPos(chrono.ChVectorD(0,0,0.0001))
         aspha.SetBodyFixed(True)
       
         aspha_asset = aspha.GetAssets()[0]
         visual_asset = chrono.CastToChVisualization(aspha_asset)
        
         vis_mat = chrono.ChVisualMaterial()
         #vis_mat.SetAmbientColor(chrono.ChColor(0,0,0))
         vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/floor.png")
         visual_asset.material_list.append(vis_mat)    
        
         system.AddBody(aspha)
            
         # Speed Limit 30 km/hr
         speedsign = chrono.ChBodyEasyCylinder(1, 0.2, 0.1, True, True, material)
         speedsign.SetPos(chrono.ChVectorD(30,-5,5))
         #speedsign.SetRot(chrono.Q_from_AngAxis(135, chrono.VECT_X))
         speedsign.SetRot(chrono.Q_from_AngAxis(3.141592/2, chrono.VECT_Z))
       
         speedsign.SetBodyFixed(True)
         speedsign_asset = speedsign.GetAssets()[0]
         visual_asset = chrono.CastToChVisualization(speedsign_asset)
       
         vis_mat = chrono.ChVisualMaterial()
       
         vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/30.png")
         visual_asset.material_list.append(vis_mat)    
         system.AddBody(speedsign)
           
    #     # Speed Limit 60 km/hr
         speedsign = chrono.ChBodyEasyCylinder(1, 0.2, 0.1, True, True, material)
         speedsign.SetPos(chrono.ChVectorD(60,-5,5))
         #speedsign.SetRot(chrono.Q_from_AngAxis(135, chrono.VECT_X))
         speedsign.SetRot(chrono.Q_from_AngAxis(3.141592/2, chrono.VECT_Z))
       
         speedsign.SetBodyFixed(True)
         speedsign_asset = speedsign.GetAssets()[0]
         visual_asset = chrono.CastToChVisualization(speedsign_asset)
         vis_mat = chrono.ChVisualMaterial()
       
         vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/60.png")
         visual_asset.material_list.append(vis_mat)    
         system.AddBody(speedsign)
       
         # Stop Sign
         speedsign = chrono.ChBodyEasyCylinder(1, 0.2, 0.1, True, True, material)
         speedsign.SetPos(chrono.ChVectorD(90,-5,5))
         #speedsign.SetRot(chrono.Q_from_AngAxis(135, chrono.VECT_X))
         speedsign.SetRot(chrono.Q_from_AngAxis(3.141592/2, chrono.VECT_Z))
       
         speedsign.SetBodyFixed(True)
         speedsign_asset = speedsign.GetAssets()[0]
         visual_asset = chrono.CastToChVisualization(speedsign_asset)
       
         vis_mat = chrono.ChVisualMaterial()
       
         vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/stop.png")
         visual_asset.material_list.append(vis_mat)    
         system.AddBody(speedsign)
           
        

        # Create and initialize the powertrain system
        powertrain = veh.ShaftsPowertrain(powertrain_file)
        self.vehicle.InitializePowertrain(powertrain)

        # Create and initialize the tires
        for axle in self.vehicle.GetAxles():
            tireL = veh.TMeasyTire(tire_file)
            self.vehicle.InitializeTire(tireL, axle.m_wheels[0], veh.VisualizationType_MESH)
            tireR = veh.TMeasyTire(tire_file)
            self.vehicle.InitializeTire(tireR, axle.m_wheels[1], veh.VisualizationType_MESH)

        # #create the driver
        self.driver = veh.ChDriver(self.vehicle)
        self.driver.Initialize()

        # initialize sensor system
        if(self.sensors):
            self.manager = sens.ChSensorManager(self.vehicle.GetSystem())
            intensity = 1.0
            self.manager.scene.AddPointLight(chrono.ChVectorF(
                2, 2.5, 100), chrono.ChVectorF(intensity, intensity, intensity), 500.0)
            b = sens.Background()
            b.mode = sens.BackgroundMode_ENVIRONMENT_MAP
            b.env_tex = chrono.GetChronoDataFile(
                "sensor/textures/quarry_01_4k.hdr")
            self.manager.scene.SetBackground(b)
            
            # AddFixedObstacles(gator.GetSystem())

            #add GPS to vehicle chassis
            offset_pose = chrono.ChFrameD(chrono.ChVectorD(-1, 0, 1), chrono.ChQuaternionD(1,0,0,0))
            gps_reference = chrono.ChVectorD(-89.400, 43.070, 260.0)
            self.gps = sens.ChGPSSensor(self.vehicle.GetChassisBody(),                     # body imu is attached to
                                10,       # update rate in Hz
                                offset_pose,             # offset pose
                                gps_reference,   #gps reference position
                                sens.ChNoiseNone()          # noise model
                                )
            self.gps.SetName("GPS Sensor")
            # Provides the host access to the gps data
            self.gps.PushFilter(sens.ChFilterGPSAccess())
            # Add the gps to the sensor manager
            self.manager.AddSensor(self.gps)

            #add magnetometer to vehicle chassis
            self.magnetometer = sens.ChMagnetometerSensor(self.vehicle.GetChassisBody(),                     # body imu is attached to
                            100,         # update rate in Hz
                            offset_pose,             # offset pose
                            sens.ChNoiseNone(),          # noise model
                            gps_reference
                            )
            self.magnetometer.SetName("Magnetometer")
            # Provides the host access to the imu data
            self.magnetometer.PushFilter(sens.ChFilterMagnetAccess())
            # Add the imu to the sensor manager
            self.manager.AddSensor(self.magnetometer)

            # create camera on vehicle hood
            cam_offset_pose = chrono.ChFrameD(chrono.ChVectorD(
                1, 0, .875), chrono.ChQuaternionD(1,0,0,0))
            self.camera = sens.ChCameraSensor(
                self.vehicle.GetChassisBody(),  # body camera is attached to
                10.0,                             # update rate in Hz
                cam_offset_pose,                    # offset pose
                1280,                           # image width
                720,                            # image height
                3.14/4                          # camera's horizontal field of view
            )
            if(self.visualize):
                self.camera.PushFilter(sens.ChFilterVisualize(640, 360, "Camera"))
            # if(self.save):
            #     if(not os.path.exists(os.path.join(self.output_dir,"camera1/"))):
            #         os.mkdir(os.path.join(self.output_dir,"camera1/"))
            #     self.camera.PushFilter(sens.ChFilterSave(os.path.join(self.output_dir,"camera1/")))
            self.camera.PushFilter(sens.ChFilterRGBA8Access())
            self.manager.AddSensor(self.camera)

            #create lidar at same location as camera
            self.lidar = sens.ChLidarSensor(
                self.vehicle.GetChassisBody(),                  # body lidar is attached to
                10,      # scanning rate in Hz
                cam_offset_pose,            # offset pose
                1000,     # number of horizontal samples
                32,       # number of vertical channels
                chrono.CH_C_PI,         # horizontal field of view
                .0,
                -.26,           # vertical field of view
                100)
            self.lidar.PushFilter(sens.ChFilterPCfromDepth())
            if(self.visualize):
                self.lidar.PushFilter(sens.ChFilterVisualizePointCloud(640, 480, 1.0, "Lidar Point Cloud"))
            # if(self.save):
            #     if(not os.path.exists(os.path.join(self.output_dir,"lidar/"))):
            #         os.mkdir(os.path.join(self.output_dir,"lidar/"))
            #     self.lidar.PushFilter(sens.ChFilterSavePtCloud(os.path.join(self.output_dir,"lidar/")))
            self.lidar.PushFilter(sens.ChFilterXYZIAccess())
            self.manager.AddSensor(self.lidar)

            # create third person camera
            cam_offset_pose2 = chrono.ChFrameD(chrono.ChVectorD(
                -12, 0, 2), chrono.Q_from_AngAxis(.1, chrono.ChVectorD(0, 1, 0)))
            self.cam2 = sens.ChCameraSensor(
                self.vehicle.GetChassisBody(),  # body camera is attached to
                60.0,                             # update rate in Hz
                cam_offset_pose2,                    # offset pose
                1280,                           # image width
                720,                            # image height
                3.14/4                          # camera's horizontal field of view
            )
            if(self.visualize):
                self.cam2.PushFilter(sens.ChFilterVisualize(640, 360, "Camera2"))
            # if(self.save):
            #     if(not os.path.exists(os.path.join(self.output_dir,"camera2/"))):
            #         os.mkdir(os.path.join(self.output_dir,"camera2/"))

            #     self.cam2.PushFilter(sens.ChFilterSave(os.path.join(self.output_dir,"camera2/")))
            self.manager.AddSensor(self.cam2)



        # add rocks randomly to the environment
        self.AddAssetRandomly(int(self.random_object_count/5),
                              chrono.GetChronoDataFile("sensor/offroad/rock1.obj"))
        self.AddAssetRandomly(int(self.random_object_count/5),
                              chrono.GetChronoDataFile("sensor/offroad/rock2.obj"))
        self.AddAssetRandomly(int(self.random_object_count/5),
                              chrono.GetChronoDataFile("sensor/offroad/rock3.obj"))
        self.AddAssetRandomly(int(self.random_object_count/5),
                              chrono.GetChronoDataFile("sensor/offroad/rock4.obj"))
        self.AddAssetRandomly(int(self.random_object_count/5),
                              chrono.GetChronoDataFile("sensor/offroad/rock5.obj"))

        # add objects from file if specified
        if(self.object_location_file != ""):
            self.AddAssetsFromFile(self.object_location_file)

        # construct scene and label rocks for generating training data
        if(self.sensors):
            self.manager.ReconstructScenes()
            self.LabelRockAssets()
            self.manager.ReconstructScenes()
            
    # ------------------------------------------------------
    # Traffic Sign Prediction - Initialsiation and Functions
    # ------------------------------------------------------
            
    # picklepath = './Trained_model.p'
    # pickle_in = open(picklepath, 'rb')
    # print(pickle_in)
    # model = pickle.load(pickle_in)
    
    # # Probability Threshold
    # threshold = 0.75
    
    # # Labelling Font
    # font = cv2.FONT_HERSHEY_TRIPLEX

    # """ Image Processing """

    # def grayscale(img):
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     return img

    # def equalize(img):
    #     img = cv2.equalizeHist(img)
    #     return img

    # def preprocessing(img):
    #     img = grayscale(img)
    #     img = equalize(img)
    #     img = img / 255
    #     return img

    # def getClassName(classNo):
    #     if classNo == 1:
    #         return '30'       # Speed Limit 30 km/hr
    #     elif classNo == 3:
    #         return '60'       # Speed Limit 30 km/hr
    #     elif classNo == 14:
    #         return '0'        # Stop
    
    
    # # Path Follower
    # path = veh.StraightLinePath(initLoc,finalLoc)
    # driver = veh.ChPathFollowerDriver(gator.GetVehicle(),path,"mypath",30)
    # temp =driver.GetSteeringController()
    # temp.SetLookAheadDistance(5)
    # temp.SetGains(0.8,0,0)
    # driver.GetSpeedController().SetGains(0.4,0,0)
    # driver.Initialize()


    def step_simulation(self):
        self.sim_step += 1
        t0 = time.time()
        pos = self.vehicle.GetPos()

        # Collect output data from modules (for inter-module communication)
        driver_inputs = self.driver.GetInputs()
        driver_inputs.m_throttle = self.throttle
        driver_inputs.m_braking = self.braking
        driver_inputs.m_steering = self.steering

        # Update modules (process inputs from other modules)
        t = self.vehicle.GetSystem().GetChTime()

        if(t > self.max_duration or (self.target_location-pos).Length() < 10.0):
            self.future.set_result(None)

            if(self.save):
                output_string = "t, PosX, PosY, PosZ, VelX, VelY, VelZ"
                np.savetxt(os.path.join(self.output_dir,"sim_data.csv"), self.output_data,header=output_string)
                self.get_logger().info("Simulation data written to '{}'.".format(os.path.join(self.output_dir,"sim_data.csv")))

            if(t > self.max_duration):
                self.get_logger().info("Simulation reached termination condition: maximum time elapsed")
            else:
                self.get_logger().info("Simulation reached termination condition: target reached")
        

        self.driver.Synchronize(t)
        self.vehicle.Synchronize(t, driver_inputs, self.terrain)
        self.terrain.Synchronize(t)

        # Advance simulation for one timestep for all modules
        self.driver.Advance(self.step_size)
        self.vehicle.Advance(self.step_size)
        self.terrain.Advance(self.step_size)

        # Update the sensor system
        if(self.sensors):
            # self.cam2.SetOffsetPose(
            #     chrono.ChFrameD(chrono.ChVectorD(-12 * np.cos(t * .5), -12 * np.sin(t * .5), 1),
            #     chrono.Q_from_AngAxis(t * .5, chrono.ChVectorD(0, 0, 1))))
            self.manager.Update()
            
            buffer = self.camera.GetMostRecentRGBA8Buffer()
            cam_data = buffer.GetRGBA8Data()
            
            # if cam_data.size != 0:
            #     imgOriginal = cv2.cvtColor(cam_data,cv2.COLOR_BGR2RGB)
            #     imgOriginal = cv2.rotate(imgOriginal,cv2.ROTATE_180)
        
            #     # Image Processing
            #     img = np.asarray(imgOriginal)
            #     img = cv2.resize(img, (32, 32))
            #     img = preprocessing(img)
            #     # cv2.imshow("Processed Image", img)
            #     img = img.reshape(1, 32, 32, 1)
            #     cv2.putText(imgOriginal, "Traffic Sign:", (20, 35), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
            #     cv2.putText(imgOriginal, "Probability:", (20, 75), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
        
            #     # Image Prediction
            #     predictions = model.predict(img)
            #     classIndex = model.predict_classes(img)
            #     probabilityValue = np.amax(predictions)
        
            #     if probabilityValue > threshold:
            #         cv2.putText(imgOriginal, str(getClassName(classIndex)), (190, 35), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
            #         cv2.putText(imgOriginal, str(round(probabilityValue*100, 2)) + "%", (180, 75), font, 0.75, (255, 0, 0), 1, cv2.LINE_AA)
        
            #         # ROS2 - To be published to the Perception node for Planning #
            #         vehicle_speed = str(getClassName(classIndex))
            #         speed = vehicle_speed
            #     else:
            #         speed = vehicle_speed
            
            


        t1 = time.time()

        t2 = time.time()
        
        # Testing Publisher and Subscription Nodes
        
        if t < 15:
            speed = 30.0;
        elif t > 15:
            speed = 60.0;
                
        # publish ROS data at correct intervals
        if(self.sim_step % self.clock_interval == 0):
            self.publish_clock()
            vehiclespeed = self.publish_vehicle_speed(speed)
        if(self.sensors):
            # if(self.sim_step % self.camera_interval == 0):
            #     self.publish_camera()
            if(self.sim_step % self.mag_interval == 0):
                self.publish_mag()
            if(self.sim_step % self.gps_interval == 0):
                self.publish_gps()
            if(self.sim_step % self.lidar_interval == 0):
                self.publish_lidar()
        if(self.publish_state_directly and (self.sim_step % self.state_interval == 0)):
            self.publish_state()
        t3 = time.time()

        #save data at 100 Hz when requested
        if(self.save and self.sim_step % self.save_interaval == 0):
            vel = self.vehicle.GetPointVelocity(chrono.ChVectorD(0.0, 0.0, 0.0))
            output = np.array([t,pos.x,pos.y,pos.z,vel.x,vel.y,vel.z])
            self.output_data.append(output)
            
        self.dur1 += t1-t0
        self.dur2 += t3-t2

        if(self.sim_step % 1000 == 0):
            self.get_logger().info("Simulation time: '%s', sim= %s, pub=%s" % (str(t),self.dur1,self.dur2))
            self.get_logger().info("Vehicle Speed: '%f'" % vehiclespeed)
            # self.get_logger().info("Vehicle pos: %s,%s,%s" % (pos.x,pos.y,pos.z))
            self.dur1 = 0
            self.dur2 = 0

    def AddAssetRandomly(self, n, filename):
        mmesh = chrono.ChTriangleMeshConnected()
        mmesh.LoadWavefrontMesh(filename, False, True)
        mmesh.Transform(chrono.ChVectorD(0, 0, 0), chrono.ChMatrix33D(1))
        for i in range(n):
            pos = chrono.ChVectorD((chrono.ChRandom(
            ) - .5) * self.terrainLength, (chrono.ChRandom() - .5) * self.terrainWidth, 0.0)
            rot = chrono.Q_from_Euler123(chrono.ChVectorD(chrono.CH_C_2PI * chrono.ChRandom(
            ), chrono.CH_C_2PI * chrono.ChRandom(), chrono.CH_C_2PI * chrono.ChRandom()))
            scale = chrono.ChVectorD(np.random.uniform(self.obstacle_scale[0],self.obstacle_scale[1]),
                np.random.uniform(self.obstacle_scale[0],self.obstacle_scale[1]),
                np.random.uniform(self.obstacle_scale[0],self.obstacle_scale[1]))
            trimesh_shape = chrono.ChTriangleMeshShape()
            trimesh_shape.SetMesh(mmesh)
            trimesh_shape.SetName(filename)
            trimesh_shape.SetStatic(True)
            trimesh_shape.SetScale(scale)
            self.rock_assets.append(trimesh_shape)
            mesh_body = chrono.ChBody()
            mesh_body.SetPos(pos)
            mesh_body.SetRot(rot)
            mesh_body.AddAsset(trimesh_shape)
            mesh_body.SetBodyFixed(True)
            self.vehicle.GetSystem().Add(mesh_body)

    def AddAssetsFromFile(self, filename):
        file_path = os.path.join(self.package_share_directory, filename)

        if(not os.path.exists(file_path)):
            self.get_logger().info("Object location file not found: '%s'" % (str(file_path)))
            exit(1)

        objects = np.loadtxt(file_path, delimiter=',')

        mesh_file = chrono.GetChronoDataFile("sensor/offroad/rock1.obj")
        mmesh = chrono.ChTriangleMeshConnected()
        mmesh.LoadWavefrontMesh(mesh_file, False, True)
        mmesh.Transform(chrono.ChVectorD(0, 0, 0), chrono.ChMatrix33D(1))

        for i in range(len(objects)):
            pos = chrono.ChVectorD(objects[i, 0], objects[i, 1], 0.0)
            rot = chrono.Q_from_Euler123(chrono.ChVectorD(chrono.CH_C_2PI * chrono.ChRandom(
            ), chrono.CH_C_2PI * chrono.ChRandom(), chrono.CH_C_2PI * chrono.ChRandom()))
            scale = chrono.ChVectorD(
                chrono.ChRandom() + .5, chrono.ChRandom() + .5, chrono.ChRandom() + .5)
            trimesh_shape = chrono.ChTriangleMeshShape()
            trimesh_shape.SetMesh(mmesh)
            trimesh_shape.SetName(mesh_file)
            trimesh_shape.SetStatic(True)
            trimesh_shape.SetScale(scale)
            self.rock_assets.append(trimesh_shape)
            mesh_body = chrono.ChBody()
            mesh_body.SetPos(pos)
            mesh_body.SetRot(rot)
            mesh_body.AddAsset(trimesh_shape)
            mesh_body.SetBodyFixed(True)
            self.vehicle.GetSystem().Add(mesh_body)

    def LabelRockAssets(self):
        rock_id = 0
        for rock in self.rock_assets:
            rock_id += 1
            for mat in rock.material_list:
                mat.SetClassID(1)
                mat.SetInstanceID(rock_id)

    # function to process data this class subscribes to
    def input_callback(self, msg):
        #settling time
        if(self.vehicle.GetSystem().GetChTime() < 2):
            return

        # clamp the step for maximum input change per time
        # braking_step = self.step_size / self.braking_time
        # throttle_step = self.step_size / self.throttle_time
        # steering_step = 2 * self.step_size / self.steering_time
        self.throttle = np.clip(
            msg.throttle, self.throttle-.1, self.throttle+.1)
        self.braking = np.clip(msg.braking, self.braking-.1, self.braking+.1)
        self.steering = np.clip(
            msg.steering, self.steering-.1, self.steering+.1)

        # clamp the input to max range
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.braking = np.clip(self.braking, 0.0, 1.0)
        self.steering = np.clip(self.steering, -.5, .5)

    # callback to run a loop and publish data this class generates
    def publish_clock(self):
        msg = Clock()
        seconds = int(self.vehicle.GetSystem().GetChTime())
        nanoseconds = int(
            (self.vehicle.GetSystem().GetChTime() - seconds) * 1e9)
        msg.clock = Time()
        msg.clock.nanosec = nanoseconds
        msg.clock.sec = seconds
        self.pub_clock.publish(msg)
        
    def publish_vehicle_speed(self,speed):
        msg = VehicleSpeed()
        msg.vehiclespeed = speed
        #self.get_logger().info('Vehicle Speed: "%f"' % msg.vehiclespeed)
        self.pub_vehiclespeed.publish(msg)
        return msg.vehiclespeed

    # def publish_camera(self):
    #     pass
        # if(self.camera and self.camera.GetMostRecentRGBA8Buffer().HasData()):
        #     img = self.camera.GetMostRecentRGBA8Buffer().GetRGBA8Data()
        #     msg = self.bridge.cv2_to_imgmsg(img, "rgba8")
        #     self.pub_image.publish(msg)

    def publish_lidar(self):
        if(self.lidar):
            data = self.lidar.GetMostRecentXYZIBuffer().GetXYZIData()
            msg = PointCloud2()
            msg.height = data.shape[0]
            msg.width = data.shape[1]
            msg.point_step = 4 #size of float
            msg.row_step = 4*4 #size of 4 floats
            msg.data = data.tobytes()
            #self.get_logger().info("lidar shape: '%s'" % (str(data.shape)))

            # msg = self.bridge.cv2_to_imgmsg(img, "rgba8")
            self.pub_lidar.publish(msg)

    def publish_mag(self):
        if(self.magnetometer):
            mag_data = self.magnetometer.GetMostRecentMagnetBuffer().GetMagnetData()

            msg = MagneticField()
            msg.magnetic_field.x = mag_data[0]
            msg.magnetic_field.y = mag_data[1]
            msg.magnetic_field.z = mag_data[2]
            self.pub_mag.publish(msg)


    def publish_gps(self):
        if(self.gps):
            gps_data = self.gps.GetMostRecentGPSBuffer().GetGPSData()

            msg = NavSatFix()
            msg.latitude = gps_data[0]
            msg.longitude = gps_data[1]
            msg.altitude = gps_data[2]
            self.pub_gps.publish(msg)

    def publish_state(self):
        pos = self.vehicle.GetPos()
        vel = self.vehicle.GetPointVelocity(
            chrono.ChVectorD(0.0, 0.0, 0.0))
        heading = self.vehicle.GetRot().Rotate(chrono.ChVectorD(1, 0, 0))

        msg = VehicleState()
        msg.position = [pos.x, pos.y, pos.z]
        msg.velocity = [vel.x, vel.y, vel.z]
        msg.heading = [heading.x, heading.y, heading.z]
        self.pub_state.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    term_condition = Future()
    sim = SimulationNode(term_condition)

    # rclpy.spin(sim)
    rclpy.spin_until_future_complete(sim,term_condition)

    sim.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
