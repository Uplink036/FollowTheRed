
#=========================import Python libs========================
import os
import numpy as np
import matplotlib.pyplot as plt

from time import sleep
#sleep(0.05)

import omni
import carb

#Start ISaac SIM with GUI
import omni.isaac
from omni.isaac.kit import SimulationApp
# import omni.isaac.utils
#=========================OPEN an Isaac SIM SESSION with GUI==========================
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1200,
    "headless": False,
    # "renderer": "RayTracedLighting",
    # "display_options": 3286,  # Set display options to show default grid
}
simulation_app = SimulationApp(CONFIG)
# simulation_app = SimulationApp({"headless": False, "additional_args": ["--/app/window/maximized=true"]})
# simulation_app = SimulationApp({"headless": False, "additional_args": ["--/app/window/resizable=true","--/app/window/maximized=true"]})
simulation_app._wait_for_viewport()

# import omni.isaac.core.utils.stage
# import omni.physx.scripts.physicsUtils
# stage = omni.isaac.core.utils.stage.get_current_stage()
# # omni.physx.scripts.physicsUtils.add_force_torque(stage,"/World/TARGET",(3,3,1))

#=========================Import OMNI Libraries=============================
import omni
from pxr import Usd, UsdGeom
import omni.usd
from omni.isaac.core import World
# import random
# import omni.isaac.debug_draw as debug_draw
#from omni.isaac.debug_draw import _debug_draw
# draw = _debug_draw.acquire_debug_draw_interface()
# import random

from omni.isaac.core.objects import DynamicCuboid, VisualCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.wheeled_robots.robots import WheeledRobot
# from omni.isaac.utils.scripts.scene_utils import create_viewport
from omni.isaac.sensor import Camera
from omni.isaac.core.utils import prims
from omni.isaac.core.articulations import Articulation


#=========================SETUP world======================
my_world = World(stage_units_in_meters=1.0)
my_world.initialize_physics()
my_world.set_simulation_dt(physics_dt=1 / 60.0, rendering_dt=1 / 60.0)

# app = Application()
# scene = simulation_app.get_scene()
#===========================Add Ground=====================
from omni.isaac.core.physics_context import PhysicsContext
PhysicsContext()
# from omni.isaac.core.objects.ground_plane import GroundPlane
# GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
ground_plane=my_world.scene.add_default_ground_plane()
#=============================Add Warehouse==================
#Add Warehouse
# assets_root_path = get_assets_root_path()
# print("ROOTPATH=\n\n\n")
# print(assets_root_path)
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.dynamic_control import _dynamic_control
# Acquire dynamic control winterface
# dc = _dynamic_control.acquire_dynamic_control_interface()
from omni.physx.scripts import utils # for warehouse collision 
# print("ROOTPATH=\n\n\n"+assets_root_path)
# wh_usdpath="C:\\Users\\Isaac\\AppData\\Local\\ov\\pkg\\isaac_sim-2023.1.1\\standalone_examples\\Warehouse01.usd"
wh_usdpath="http://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/ArchVis/Industrial/Buildings/Warehouse/Warehouse01.usd"
wh_path = "/World/Warehouse"
wh_prim = add_reference_to_stage(usd_path=wh_usdpath, prim_path=wh_path)
xformAPI = UsdGeom.XformCommonAPI(wh_prim)
xformAPI.SetScale((0.001, 0.001, 0.001))
utils.setCollider(wh_prim) #adds collider to warehouse
#===========================Add Jetbot======================
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
JB = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0.5, 0.5, 0.1]),
    )
)
# JBprim = my_world.stage.GetPrimAtPath("/World/Jetbot")
# xformAPI = UsdGeom.XformCommonAPI(JBprim)
# xformAPI.SetScale((10, 10, 10))
#---------Add differential controller to the robot----
JB_controller = DifferentialController(name="simple_control", wheel_radius=0.03, wheel_base=0.1125)
# ----------Get the camera object------------
camera_prim_path = "/World/Jetbot/chassis/rgb_camera/jetbot_camera"
JBcam = Camera(camera_prim_path)
JBcam.initialize()
#==============Add Obstacles=============
OBJ=[]
Nobj=20
Posbounds=np.array([[-3, 3],[-3, 3],[.1, .2]])
Scalebounds=np.array([[.1,.2],[.1,.2],[.1,.2]])

for i in range(Nobj):
    x=np.random.rand()*(Posbounds[0][0]-Posbounds[0][1])+Posbounds[0][1]
    y=np.random.rand()*(Posbounds[1][0]-Posbounds[1][1])+Posbounds[1][1]
    z=np.random.rand()*(Posbounds[2][0]-Posbounds[2][1])+Posbounds[2][1]
    xscale=np.random.rand()*(Scalebounds[0][0]-Scalebounds[0][1])+Scalebounds[0][1]
    yscale=np.random.rand()*(Scalebounds[1][0]-Scalebounds[1][1])+Scalebounds[1][1]
    zscale=np.random.rand()*(Scalebounds[2][0]-Scalebounds[2][1])+Scalebounds[2][1]
    if i==0:
        objname="TARGET"
        objcolor=np.array([255, 0, 0])
        xscale=.2
        yscale=.2
        zscale=.2
        x=.1
        y=0
        z=0
    else:
        objname="Obstacle"+str(i)
        objcolor=np.array([0, 0, 255])
        xscale=np.random.rand()*(Scalebounds[0][0]-Scalebounds[0][1])+Scalebounds[0][1]
        yscale=np.random.rand()*(Scalebounds[1][0]-Scalebounds[1][1])+Scalebounds[1][1]
        zscale=np.random.rand()*(Scalebounds[2][0]-Scalebounds[2][1])+Scalebounds[2][1]

    OBJ.append(
        my_world.scene.add(
        DynamicCuboid(
            prim_path="/World/"+objname,
            name=objname,
            position=np.array([x, y, z]),
            scale=np.array([xscale, yscale, zscale]),
            size=1.0,
            color=objcolor,
            linear_velocity=np.array([0, 0, 1]),
        )
    )
    )
#=================Initiate Obstacles Movement=====================
# curtime=0
# movdim=0
# movdir=0
minmovestep=.001
maxmovestep=.05
minmovedur=1 #Moving duration
maxmovedur=4
OBJ_movesteps=np.random.rand(Nobj)*(maxmovestep-minmovestep)+minmovestep;
OBJ_movedurs=np.random.rand(Nobj)*(maxmovedur-minmovedur)+minmovedur;
OBJ_curtimes=np.zeros(Nobj)
OBJ_movedims=np.zeros(Nobj)
OBJ_movedirs=np.ones(Nobj)
OBJ_pos=np.zeros([Nobj,3])

#=========Draw Line=============
start_point = [(0,0,0)]  # Adjust these values for your desired start position
end_point = [(0,0,3)]  # Adjust these values for your desired end p
line_color = [(1.0, 0.0, 0.0, 1.0)]  # Red color (RGBA format)
line_width = [2.0]  # Line thickness
#draw.draw_lines(start_point, end_point, line_color, line_width)
#=====================START SIMULATION=====================
disp_time=.1
curtime=0
my_world.reset()
direction = 1
speed = 1

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_playing():
        if my_world.current_time_step_index == 0:
            my_world.reset()
            JB_controller.reset()
        # print('step'+str(my_world.current_time_step_index)+' time=' + str(my_world.current_time) + '\n')
        #=====================================RUNTIME======================================================
        #---------Move Obstacles------------
        for i in range(Nobj):
            #----------Update Moves----------
            if my_world.current_time>(OBJ_curtimes[i]+OBJ_movedurs[i]):
                OBJ_curtimes[i]=my_world.current_time
                OBJ_movesteps[i]=np.random.rand()*(maxmovestep-minmovestep)+minmovestep
                OBJ_movedurs[i]=np.random.rand()*(maxmovedur-minmovedur)+minmovedur
                OBJ_movedims[i]=np.random.randint(2)#X or Y
                OBJ_movedirs[i]=2*np.random.randint(2)-1#increase/decrease
                # print("Changing direction: \n")
            #----------MOVE positions-------------
            pos=OBJ[i].get_world_pose()[0]  
            OBJ_pos[i]=pos
            x=OBJ_pos[i][0]
            y=OBJ_pos[i][1]
            z=OBJ_pos[i][2]
            if OBJ_curtimes[i]>0:                
                if x>3 or y>3 or x<-3 or y<-3:
                    OBJ_movedirs[i]=-OBJ_movedirs[i]
                x+=(1-OBJ_movedims[i])*OBJ_movedirs[i]*OBJ_movesteps[i]
                y+=OBJ_movedims[i]*OBJ_movedirs[i]*OBJ_movesteps[i]
                z+=.001
                OBJ[i].set_world_pose(position=np.array([x, y, z]), orientation=np.array([1., 0., 0., 0.]))

        #-------------GET JB Camera----------------
        CAMFRAME=JBcam.get_current_frame()
        CAMRGB=JBcam.get_rgba()
        # print("Camera Frame:\n"+str(CAMFRAME['rgba']))
        # print("Camera Frame:\n"+str(CAMFRAME['rendering_frame']))
        #print("Camera RGB:\n"+str(CAMRGB.shape))
        #-----------PROCESS JB RGB Image-----------
        red_areas=np.all([CAMRGB[:,:,0]>200,CAMRGB[:,:,1]<100,CAMRGB[:,:,2]<100],axis=0)
        
        red_cols=np.argwhere(np.any(red_areas, axis=0))
        red_size=len(red_cols)
        red_loc=0
        if red_size>0:
            red_loc=-1+max(red_cols)/64
        

        blue_areas=np.all([CAMRGB[:,:,2]>250,CAMRGB[:,:,0]<100,CAMRGB[:,:,1]<100],axis=0)
        blue_rows=np.argwhere(np.any(blue_areas, axis=0))
        blue_size=len(blue_rows)
        # blue_centered=blue_rows-64*np.ones(len(blue_rows))
        # center_ind=np.where(blue_centered == blue_centered.min())
        # blue_diffs=[1]+np.diff(blue_rows)
        # center_starts=np.where(blue_diffs>2)
        # center_starts=center_starts-center_ind*np.ones(len(center_starts))
        # blue_srart=

        if my_world.current_time>curtime:
            curtime=my_world.current_time+disp_time
            # print(red_size)
            # if red_size>0:
            #     print(red_loc,"\n")
        #print(f"SIZE={red_size}\tLR={red_loc}")
        # #=========Draw Line=============
        # start_point = [(0,0,0)]  # Adjust these values for your desired start position
        # end_point = [(0,0,3)]  # Adjust these values for your desired end p
        # line_color = [(1.0, 0.0, 0.0, 1.0)]  # Red color (RGBA format)
        # line_width = [2.0]  # Line thickness
        # draw.draw_lines(start_point, end_point, line_color, line_width)
        #----------------JETBOT Movement---------------
        position, orientation = JB.get_world_pose()
        # JB.apply_wheel_actions(JB_controller.forward(command=[1, np.pi]))
        
        # JB.apply_wheel_actions(JB_controller.forward(command=[0, np.pi]))
        image_half_width = CAMRGB.shape[0]/2
        if red_size > 0:
            print(f"{red_cols=}")
            mean_loc = np.mean(red_cols)
            speed = abs(mean_loc-image_half_width)/image_half_width
            if np.mean(red_cols) > image_half_width:
                direction = -1
            elif np.mean(red_cols) < image_half_width:
                direction = 1
        JB.apply_wheel_actions(JB_controller.forward(command=[0, direction* speed*np.pi]))

        #print(JB.get_angular_velocity())
        
        
        #=====================================END of RUNTIME======================================================
        if my_world.current_time_step_index == 0:
            my_world.reset()
            # my_controller.reset()
        observations = my_world.get_observations()
simulation_app.close()
