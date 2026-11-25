"""Colmibot simplificado en Gazebo, pensado para la VM de Parallels.

Usa colmibot_gazebo.urdf: mismos joints, DiffDrive y sensores (IMU, cámara,
ToF x3) que robot_description.urdf, pero con primitivas (cajas, cilindros,
esferas) en lugar de mallas. Los tópicos son los mismos que con el modelo
completo, así que box_detector y ruta_escenario1 funcionan sin cambios.

Argumentos:
  mundo:=prueba|escenario1   mundo a cargar               [prueba]
  control:=true|false        joystick + colmibot_controller [true]
                             (usar false con ruta_escenario1, porque ambos
                             publican en /cmd_vel)

Ejemplos:
  ros2 launch robot_description gazebo_simple.launch.py
  ros2 launch robot_description gazebo_simple.launch.py mundo:=escenario1 control:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Pose inicial por mundo: (x, y, z, yaw). Coincide con los otros launch.
MUNDOS = {
    "prueba": ("prueba.sdf", ("-1.0", "0.0", "0.1", "0.0")),
    "escenario1": ("escenario1.sdf", ("6.5", "4.5", "0.1", "-1.5708")),
}


def lanzar_gazebo(context):
    pkg = get_package_share_directory("robot_description")
    mundo = LaunchConfiguration("mundo").perform(context)
    if mundo not in MUNDOS:
        raise RuntimeError(f"mundo '{mundo}' no válido; opciones: {list(MUNDOS)}")

    world_file, (x, y, z, yaw) = MUNDOS[mundo]
    world_path = os.path.join(pkg, "worlds", world_file)
    robot_file = os.path.join(pkg, "urdf", "colmibot_gazebo.urdf")

    return [
        # Servidor de Gazebo (física + render de sensores con ogre2, según el mundo)
        ExecuteProcess(
            cmd=["gz", "sim", "-s", "-r", "-v", "3", world_path],
            output="screen",
        ),
        # GUI por separado con ogre2 (ogre 1.9 truena con sensores de render en la VM)
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=["gz", "sim", "-g", "-v", "3", "--render-engine", "ogre2"],
                    output="screen",
                )
            ],
        ),
        # Insertar el Colmibot simplificado
        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package="ros_gz_sim",
                    executable="create",
                    output="screen",
                    arguments=[
                        "-file", robot_file,
                        "-name", "colmibot",
                        "-x", x, "-y", y, "-z", z, "-Y", yaw,
                    ],
                )
            ],
        ),
    ]


def generate_launch_description():
    pkg = get_package_share_directory("robot_description")
    robot_file = os.path.join(pkg, "urdf", "colmibot_gazebo.urdf")
    bridge_config = os.path.join(pkg, "config", "gz_bridge.yaml")

    with open(robot_file) as f:
        robot_description = f.read()

    control = LaunchConfiguration("control")

    return LaunchDescription([
        DeclareLaunchArgument("mundo", default_value="prueba",
                              description="prueba | escenario1"),
        DeclareLaunchArgument("control", default_value="true",
                              description="Joystick + colmibot_controller"),

        # Configuración para Parallels (Mac ARM): render por software.
        # Sin GALLIUM_DRIVER y MESA_LOADER_DRIVER_OVERRIDE, Mesa abre por EGL
        # el driver de la GPU virtual de Parallels y el servidor truena con
        # "Segmentation fault" en driCreateNewScreen3 al iniciar el render.
        SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
        SetEnvironmentVariable("GALLIUM_DRIVER", "llvmpipe"),
        SetEnvironmentVariable("MESA_LOADER_DRIVER_OVERRIDE", "kms_swrast"),
        SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
        # Para que Gazebo encuentre model://pallet y model://rescue_randy (prueba.sdf)
        AppendEnvironmentVariable("GZ_SIM_RESOURCE_PATH", os.path.join(pkg, "models")),

        OpaqueFunction(function=lanzar_gazebo),

        # Publica el URDF y calcula /tf
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description, "use_sim_time": True}],
        ),

        # Puente ROS 2 <-> Gazebo (clock, cmd_vel, odom, imu, cámara, ToF)
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            output="screen",
            parameters=[{"config_file": bridge_config, "use_sim_time": True}],
        ),

        # Joystick PlayStation: solo mueve mientras se presiona X
        Node(package="joy", executable="joy_node", output="screen",
             condition=IfCondition(control)),
        Node(package="colmibotbasics", executable="colmibot_controller", output="screen",
             condition=IfCondition(control)),
    ])
