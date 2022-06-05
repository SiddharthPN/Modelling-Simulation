# Author: Siddharth Palani Natarajan 

# ME 468 - Traffic Sign Detection for Autonomous Vehicle

#########################################################

# =============================================================================
# PROJECT CHRONO - http:#projectchrono.org
#
# Copyright (c) 2014 projectchrono.org
# All right reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http:#projectchrono.org/license-chrono.txt.
#
# =============================================================================

import pychrono.core as chrono
import pychrono.vehicle as veh
import pychrono.sensor as sens
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

# =============================================================================

# The path to the Chrono data directory containing various assets (meshes, textures, data files)
# is automatically set, relative to the default location of this demo.
# If running from a different directory, you must change the path to the data directory with: 
chrono.SetChronoDataPath(os.getenv("CHRONO_DATA_DIR"))
veh.SetDataPath(chrono.GetChronoDataPath() + 'vehicle/')


# ---------------------
# Simulation parameters
# ---------------------

# Initial vehicle location and orientation
initLoc = chrono.ChVectorD(0, 0, 0.4)
initRot = chrono.ChQuaternionD(1, 0, 0, 0)
finalLoc = chrono.ChVectorD(100, 0, 0.4)
# Visualization type for vehicle parts (PRIMITIVES, MESH, or NONE)
chassis_vis_type = veh.VisualizationType_MESH
suspension_vis_type = veh.VisualizationType_PRIMITIVES
steering_vis_type = veh.VisualizationType_PRIMITIVES
wheel_vis_type = veh.VisualizationType_NONE
tire_vis_type = veh.VisualizationType_MESH

# Simulation step sizes
step_size = 1e-3
tire_step_size = step_size

# Simulation end time
tend = 30

# view camera images
vis = True
save = False
noise = False

# =============================================================================

# =============================================================================
# =============================================================================
# Create signs

def AddFixedObstacles(system) :
    # Create contact material, of appropriate type. Use default properties
    matNSC = chrono.ChMaterialSurfaceNSC()
    matSMC = chrono.ChMaterialSurfaceSMC()
    material = matNSC
    material2 = matSMC
    
    #########asphalt
    aspha = chrono.ChBodyEasyBox(10000,2,1,0.1,True,False,material2)
    aspha.SetPos(chrono.ChVectorD(0,0,0.0001))
    aspha.SetBodyFixed(True)

    aspha_asset = aspha.GetAssets()[0]
    visual_asset = chrono.CastToChVisualization(aspha_asset)

    vis_mat = chrono.ChVisualMaterial()
    vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/floor.png")
    visual_asset.material_list.append(vis_mat)    

    system.AddBody(aspha)
    
    #####speed signs 30
    speedsign = chrono.ChBodyEasyCylinder(1, 0.2, 0.1, True, True, material)
    speedsign.SetPos(chrono.ChVectorD(30,-5,5))
    speedsign.SetRot(chrono.Q_from_AngAxis(3.141592/2, chrono.VECT_Z))

    speedsign.SetBodyFixed(True)
    speedsign_asset = speedsign.GetAssets()[0]
    visual_asset = chrono.CastToChVisualization(speedsign_asset)

    vis_mat = chrono.ChVisualMaterial()

    vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/30.png")
    visual_asset.material_list.append(vis_mat)    
    system.AddBody(speedsign)
   
    #####speed signs 60
    speedsign = chrono.ChBodyEasyCylinder(1, 0.2, 0.1, True, True, material)
    speedsign.SetPos(chrono.ChVectorD(60,-5,5))
    speedsign.SetRot(chrono.Q_from_AngAxis(3.141592/2, chrono.VECT_Z))

    speedsign.SetBodyFixed(True)
    speedsign_asset = speedsign.GetAssets()[0]
    visual_asset = chrono.CastToChVisualization(speedsign_asset)

    vis_mat = chrono.ChVisualMaterial()

    vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/60.png")
    visual_asset.material_list.append(vis_mat)    
    system.AddBody(speedsign)



    #####speed signs 0
    speedsign = chrono.ChBodyEasyCylinder(1, 0.2, 0.1, True, True, material)
    speedsign.SetPos(chrono.ChVectorD(90,-5,5))
    speedsign.SetRot(chrono.Q_from_AngAxis(3.141592/2, chrono.VECT_Z))

    speedsign.SetBodyFixed(True)
    speedsign_asset = speedsign.GetAssets()[0]
    visual_asset = chrono.CastToChVisualization(speedsign_asset)

    vis_mat = chrono.ChVisualMaterial()

    vis_mat.SetKdTexture("/srv/home/yrhee6/ME468/PJ/textures/stop.png")
    visual_asset.material_list.append(vis_mat)    
    system.AddBody(speedsign)
    

    











print( "Copyright (c) 2017 projectchrono.org\n")

# --------------
# Vehicle System
# --------------

# Create the vehicle, set parameters, and initialize
gator = veh.Gator()
gator.SetContactMethod(chrono.ChContactMethod_NSC)
gator.SetChassisFixed(False)
gator.SetInitPosition(chrono.ChCoordsysD(initLoc, initRot))
gator.SetBrakeType(veh.BrakeType_SHAFTS)
gator.SetTireType(veh.TireModelType_TMEASY)
gator.SetTireStepSize(tire_step_size)
gator.SetInitFwdVel(0.0)
gator.Initialize()

gator.SetChassisVisualizationType(chassis_vis_type)
gator.SetSuspensionVisualizationType(suspension_vis_type)
gator.SetSteeringVisualizationType(steering_vis_type)
gator.SetWheelVisualizationType(wheel_vis_type)
gator.SetTireVisualizationType(tire_vis_type)

# ------------------
# Terrain
# ------------------
terrain = veh.RigidTerrain(gator.GetSystem())
patch_mat = chrono.ChMaterialSurfaceNSC()
patch_mat.SetFriction(0.9)
patch_mat.SetRestitution(0.01)
patch = terrain.AddPatch(patch_mat, 
                         chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(0, 0, 1), 
                         600, 600)
patch.SetColor(chrono.ChColor(0.8, 0.8, 1.0))
patch.SetTexture(veh.GetDataFile("terrain/textures/tile4.jpg"), 1200, 1200)
terrain.Initialize()

asset = patch.GetGroundBody().GetAssets()[0]
visual_asset = chrono.CastToChVisualization(asset)

vis_mat = chrono.ChVisualMaterial()
vis_mat.SetKdTexture(chrono.GetChronoDataFile("sensor/textures/mud.png"))
vis_mat.SetRoughness(0.99)

visual_asset.material_list.append(vis_mat)

# -----------------------
# Sensor Manager
# -----------------------
manager = sens.ChSensorManager(gator.GetSystem())
intensity = 1.0
manager.scene.AddPointLight(chrono.ChVectorF(0, 0, 100), chrono.ChVectorF(intensity, intensity, intensity), 500.0)
b = sens.Background()
b.mode = sens.BackgroundMode_ENVIRONMENT_MAP
b.env_tex = chrono.GetChronoDataFile("sensor/textures/quarry_01_4k.hdr")
manager.scene.SetBackground(b)


##AddObstacles
AddFixedObstacles(gator.GetSystem())


# ------------------------------------------------
# Camera Sensor
# ------------------------------------------------

offset_pose = chrono.ChFrameD(chrono.ChVectorD(.1, 0, 1.45), chrono.Q_from_AngAxis(.2, chrono.ChVectorD(0, 1, 0)))

# Camera Sensor Parameters
cam_update_rate = 10
image_width = 1280
image_height = 720
fov = 1.5707963268
cam_lag = 0
cam_collection_time = 1. / float(cam_update_rate)

out_dir = "SENSOR_OUTPUT/SENSORS_PY"

cam = sens.ChCameraSensor(
    gator.GetChassisBody(),
    cam_update_rate,
    offset_pose,
    image_width,
    image_height,
    fov
)

cam.SetName("Camera Sensor")
cam.SetLag(cam_lag)
cam.SetCollectionWindow(cam_collection_time)

cam.PushFilter(sens.ChFilterRGBA8Access())
cam.PushFilter(sens.ChFilterSave(out_dir + "/rgb/"))

manager.AddSensor(cam)


# ------------------------------------------------
# GPS Sensor
# ------------------------------------------------

# GPS Sensor Parameters
gps_update_rate = 10
gps_reference = chrono.ChVectorD(-89.400, 43.070, 260.0)
gps_lag = 0
gps_collection_time = 0
noise_none = sens.ChNoiseNone()

offset_pose = chrono.ChFrameD(chrono.ChVectorD(.1, 0, 1.45), chrono.Q_from_AngAxis(.2, chrono.ChVectorD(0, 1, 0)))

gps = sens.ChGPSSensor(
    gator.GetChassisBody(),
    gps_update_rate,
    offset_pose,
    gps_reference,
    noise_none
)

gps.SetName("GPS Sensor")
gps.SetLag(gps_lag)
gps.SetCollectionWindow(gps_collection_time)
gps.PushFilter(sens.ChFilterGPSAccess())

manager.AddSensor(gps)


# ------------------------------------------------------
# Traffic Sign Prediction - Initialsiation and Functions
# ------------------------------------------------------

# #Import Trained
# picklepath = './Trained_model.p'
# pickle_in = open(picklepath, 'rb')
# print('Check1')
# print(pickle_in)
# print('Check2')
# model = pickle.load(pickle_in)

# Probability Threshold
threshold = 0.75

# Labelling Font
font = cv2.FONT_HERSHEY_TRIPLEX

""" Image Processing """

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img

def getClassName(classNo):
    if classNo == 1:
        return '30'  # Speed Limit 30 km/hr
    elif classNo == 3:
        return '60'  # Speed Limit 30 km/hr
    elif classNo == 14:
        return '0'  # STOP
    
    

#create path follower
path = veh.StraightLinePath(initLoc,finalLoc)
driver = veh.ChPathFollowerDriver(gator.GetVehicle(),path,"mypath",45)
temp =driver.GetSteeringController()
temp.SetLookAheadDistance(5)
temp.SetGains(0.8,0,0)
driver.GetSpeedController().SetGains(0.4,0,0)

driver.Initialize()
# ---------------
# Simulation loop
# ---------------
time = 0.0

while (time < tend) :
    time = gator.GetSystem().GetChTime()



    # Collect output data from modules (for inter-module communication)

    driver_inputs = driver.GetInputs()


    ###This is just to show changing speed works########################
    currentspeed  = driver.GetSpeedController().GetCurrentSpeed()

    if currentspeed >30 and time >tend/4:
        driver_inputs.m_braking = 0.5
        driver_inputs.m_throttle = driver_inputs.m_throttle -0.3
    else:
        driver_intputs = driver.GetInputs()



    terrain.Synchronize(time)
    gator.Synchronize(time, driver_inputs, terrain)
    driver.Synchronize(time)
    
    manager.Update()
    
    buffer = cam.GetMostRecentRGBA8Buffer()
    cam_data = buffer.GetRGBA8Data()
        
    if cam_data.size != 0:
        imgOriginal = cv2.cvtColor(cam_data,cv2.COLOR_BGR2RGB)
        imgOriginal = cv2.rotate(imgOriginal,cv2.ROTATE_180)
        
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
    #         print(vehicle_speed)
    #     else:
    #         print(vehicle_speed)


    # Advance simulation for one timestep for all modules
    terrain.Advance(step_size)
    gator.Advance(step_size)
    driver.Advance(step_size)

