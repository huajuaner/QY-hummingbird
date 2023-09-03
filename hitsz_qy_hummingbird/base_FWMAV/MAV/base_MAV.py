import time
from hitsz_qy_hummingbird.base_FWMAV.MAV.MAV_params import ParamsForBaseMAV
from hitsz_qy_hummingbird.configuration.configuration import GLOBAL_CONFIGURATION
import numpy as np
import pybullet as p
import pybullet_data


class BaseMAV:
    """
    This class should have as few functions as possible,
    concentrating all PyBullet environment functionalities internally.
    """

    def __init__(self,
                 urdf_name: str,
                 mav_params: ParamsForBaseMAV,
                 if_gui: bool,
                 if_fixed: bool):
        """
        the urdf should be loaded and the variables should be set up
        the joints and rods are defined accordingly,
        and the configuration should be set up.
        """

        self.params = mav_params
        self.params.initial_orientation = p.getQuaternionFromEuler(self.params.initial_rpy)
        self.data = {}
        self.logger = GLOBAL_CONFIGURATION.logger

        self.right_stroke_joint = self.params.right_stroke_joint
        self.right_rotate_joint = self.params.right_rotate_joint
        self.left_stroke_joint = self.params.left_stroke_joint
        self.left_rotate_joint = self.params.left_rotate_joint
        self.right_rod_link = self.params.right_rod_link
        self.left_rod_link = self.params.left_rod_link
        self.right_wing_link = self.params.right_wing_link
        self.left_wing_link = self.params.left_wing_link

        self.right_stroke_amp = 0
        self.right_stroke_vel = 0
        self.right_stroke_acc = 0
        self.target_right_stroke_amp = 0
        self.target_right_stroke_vel = 0
        self.target_right_stroke_acc = 0

        self.left_stroke_amp = 0
        self.left_stroke_vel = 0
        self.left_stroke_acc = 0
        self.target_left_stroke_amp = 0
        self.target_left_stroke_vel = 0
        self.target_left_stroke_acc = 0

        self.right_rotate_amp = 0
        self.right_rotate_vel = 0

        self.left_rotate_amp = 0
        self.left_rotate_vel = 0

        if if_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        #COV_ENABLE_RGB_BUFFER_PREVIEW：Enable or disable RGB buffer preview.
        #COV_ENABLE_DEPTH_BUFFER_PREVIEW：Enable or disable depth buffer preview.
        #COV_ENABLE_SEGMENTATION_MARK_PREVIEW：Enable or disable segmentation mark preview.
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        p.setGravity(self.params.gravity[0],
                     self.params.gravity[1],
                     self.params.gravity[2])

        p.loadURDF("plane.urdf")

        # URDF_USE_INERTIA_FROM_FILE：By default, Bullet recalculates the inertia tensor based on the mass and volume of the collision shape. 
        # If a more accurate inertia tensor can be provided, use this flag.
        self.body_unique_id = p.loadURDF(fileName=urdf_name,
                                         basePosition=self.params.initial_xyz,
                                         baseOrientation=self.params.initial_orientation,
                                         flags=p.URDF_USE_INERTIA_FROM_FILE)
        # local inertia diagonal
        # getDynamicsInfovec3[2]：list of 3 floats
        # Local inertia diagonal. Please note that the link and base are centered on the center of mass and aligned with the principal axes of inertia. 
        # The inertia tensor describes the inertial characteristics of an object when it rotates around its center of mass.
        # It is a symmetric matrix that can be diagonalized, that is, a coordinate system can be found such that the representation of the inertia tensor in that coordinate system is a diagonal matrix. 
        # The diagonal elements of this diagonal matrix are the local inertia diagonals.
        self.if_valid_urdf = True
        for linkid in range(p.getNumJoints(self.body_unique_id)):
            inertia = p.getDynamicsInfo(self.body_unique_id, linkid)[2]
            if sum(inertia) == 0:
                self.if_valid_urdf = False

        # stepSimulation performs all operations, such as collision detection, constraint solving, and integration, in a single forward dynamics simulation step. 
        # The default time step is 1/240 seconds, which can be changed using the setTimeStep or setPhysicsEngineParameter API.
        # The number of solver iterations and the error reduction parameters (erp) for contact, friction, and non-contact joints are related to the time step. 
        # If you change the time step, you may need to adjust these values accordingly, especially the erp values.
        p.setTimeStep(1 / GLOBAL_CONFIGURATION.TIMESTEP)

        # URDF, SDF, and MJCF specify articulated bodies as acyclic tree structures. ‘createConstraint’ allows connecting specific links of the main body to form a closed loop. 
        # Arbitrary constraints can also be created between objects and between objects and a specific world frame.
        # createConstraint returns an integer unique id that can be used to change or delete the constraint.
        self.if_fixed = if_fixed
        if self.if_fixed:
            self.constraint_id = p.createConstraint(self.body_unique_id,
                                                    parentLinkIndex=-1,
                                                    childBodyUniqueId=-1,
                                                    childLinkIndex=-1,
                                                    jointType=p.JOINT_FIXED,
                                                    jointAxis=[0, 0, 0],
                                                    parentFramePosition=[0, 0, 0],
                                                    childFramePosition=self.params.initial_xyz,
                                                    childFrameOrientation=self.params.initial_orientation)
        
        self.camera_follow()

        #Use changeDynamics to change properties.
        #change the lower limit of a joint, also requires
        #jointUpperLimit otherwise it is ignored. note that at
        #the moment, the joint limits are not updated in
        #'getJointInfo'!
        p.changeDynamics(self.body_unique_id,
                         self.right_wing_link,
                         jointLowerLimit=-1 * self.params.max_angle_of_rotate,
                         jointUpperLimit=self.params.max_angle_of_rotate)

        p.changeDynamics(self.body_unique_id,
                         self.right_wing_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)

        p.changeDynamics(self.body_unique_id,
                         self.left_wing_link,
                         jointLowerLimit=-1 * self.params.max_angle_of_rotate,
                         jointUpperLimit=self.params.max_angle_of_rotate)

        p.changeDynamics(self.body_unique_id,
                         self.left_wing_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)

        # p.changeDynamics(self.body_unique_id,
        #                  self.right_rod_link,
        #                  jointLowerLimit=-1 * self.params.max_angle_of_stroke,
        #                  jointUpperLimit=self.params.max_angle_of_stroke)

        p.changeDynamics(self.body_unique_id,
                         self.right_rod_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)

        # p.changeDynamics(self.body_unique_id,
        #                  self.right_rod_link,
        #                  jointLowerLimit=-1 * self.params.max_angle_of_stroke,
        #                  jointUpperLimit=self.params.max_angle_of_stroke,)

        p.changeDynamics(self.body_unique_id,
                         self.left_rod_link,
                         lateralFriction=0,
                         linearDamping=0,
                         angularDamping=0,
                         maxJointVelocity=self.params.max_joint_velocity)
        
        p.setJointMotorControl2(self.body_unique_id,
                                self.right_stroke_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

        p.setJointMotorControl2(self.body_unique_id,
                                self.left_stroke_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

        p.setJointMotorControl2(self.body_unique_id,
                                self.right_rotate_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

        p.setJointMotorControl2(self.body_unique_id,
                                self.left_rotate_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                force=0)

    def camera_follow(self):
        """
        the camera should be always targeted at the MAV
        """
        position, orientation = p.getBasePositionAndOrientation(self.body_unique_id)
        p.resetDebugVisualizerCamera(cameraDistance=0.3,
                                     cameraYaw=120,
                                     cameraPitch=-20,
                                     cameraTargetPosition=position)

    def step(self):
        """
        The torque and the force are applied, and the influence is calculated
        the
        """
        GLOBAL_CONFIGURATION.TICKTOCK = GLOBAL_CONFIGURATION.TICKTOCK + 1
        p.stepSimulation()
        self.camera_follow()
        time.sleep(self.params.sleep_time)
        self.joint_state_update()

    def joint_state_update(self):
        """
        The joint state should be renewed
        """
        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.right_stroke_joint)
        self.right_stroke_amp = pos
        self.right_stroke_acc = (vel - self.right_stroke_vel) * GLOBAL_CONFIGURATION.TIMESTEP
        self.right_stroke_vel = vel

        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.left_stroke_joint)
        self.left_stroke_amp = pos
        self.left_stroke_acc = (vel - self.left_stroke_vel) * GLOBAL_CONFIGURATION.TIMESTEP
        self.left_stroke_vel = vel

        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.right_rotate_joint)
        self.right_rotate_amp = pos
        self.right_rotate_vel = vel

        pos, vel, _, _ = p.getJointState(self.body_unique_id,
                                         self.left_rotate_joint)
        self.left_rotate_amp = pos
        self.left_rotate_vel = vel

    def joint_control(self,
                      target_right_stroke_amp=None,
                      target_right_stroke_vel=None,
                      target_left_stroke_amp=None,
                      target_left_stroke_vel=None,
                      right_input_torque=None,
                      left_input_torque=None):
        """
        the motor is controlled according to the control mode
        """
        #The practical implementation of the motor controller for POSITION_CONTROL and VELOCITY_CONTROL is used as a constraint. 
        # The parameters positionGain and velocityGain are used to minimize the error, error =position_gain*(desired_position-actual_position)+velocity_gain*(desired_velocity-actual_velocity)
        if target_right_stroke_amp is not None and target_right_stroke_vel is not None and target_left_stroke_amp is not None and target_left_stroke_vel is not None:

            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_right_stroke_amp,
                                    targetVelocity=target_right_stroke_vel,
                                    force=self.params.max_force,
                                    positionGain=self.params.position_gain,
                                    velocityGain=self.params.velocity_gain,
                                    maxVelocity=self.params.max_joint_velocity)

            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_left_stroke_amp,
                                    targetVelocity=target_left_stroke_vel,
                                    force=self.params.max_force,
                                    positionGain=self.params.position_gain,
                                    velocityGain=self.params.velocity_gain,
                                    maxVelocity=self.params.max_joint_velocity)

        elif target_right_stroke_vel is not None and target_left_stroke_vel is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=target_right_stroke_vel,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)
            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=target_left_stroke_vel,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)

        elif right_input_torque is not None and left_input_torque is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=right_input_torque,
                                    maxVelocity=self.params.max_joint_velocity)
            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.TORQUE_CONTROL,
                                    force=left_input_torque,
                                    maxVelocity=self.params.max_joint_velocity)
        elif target_right_stroke_amp is not None and target_left_stroke_amp is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_right_stroke_amp,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)

            p.setJointMotorControl2(self.body_unique_id,
                                    self.left_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_left_stroke_amp,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)
        elif target_right_stroke_amp is not None:
            p.setJointMotorControl2(self.body_unique_id,
                                    self.right_stroke_joint,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_right_stroke_amp,
                                    force=self.params.max_force,
                                    maxVelocity=self.params.max_joint_velocity)

    def get_state_for_motor_torque(self):
        """
        return the  right_stroke_amp    right_stroke_vel    right_stroke_acc    right_torque
                    left_stroke_amp     left_stroke_vel     left_stroke_acc     left_torque
        """
        #getJointState[3]: appliedJointMotorTorque
        right_torque = p.getJointState(self.body_unique_id,
                                       self.right_stroke_joint)[3]
        left_torque = p.getJointState(self.body_unique_id,
                                      self.left_stroke_joint)[3]
        return (self.right_stroke_amp,
                self.right_stroke_vel,
                self.right_stroke_acc,
                right_torque,
                self.left_stroke_amp,
                self.left_stroke_vel,
                self.left_stroke_acc,
                left_torque)

    def get_state_for_wing(self):
        """
        return the  right_stroke_angular_velocity   right_rotate_angular_velocity
                    right_c_axis        right_r_axis        right_z_axis
                    left_stroke_angular_velocity    left_rotate_angular_velocity
                    left_c_axis         left_r_axis         left_z_axis
        """
        #getLinkState[7]：worldLinkAngularVelocity，vec3, list of 3 floats
        right_sum_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.right_wing_link,
                           computeLinkVelocity=1)[7])
        right_stroke_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.right_rod_link,
                           computeLinkVelocity=1)[7])
        right_rotate_angular_velocity = right_sum_angular_velocity - right_stroke_angular_velocity

        left_sum_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.left_wing_link,
                           computeLinkVelocity=1)[7])
        left_stroke_angular_velocity = np.array(
            p.getLinkState(self.body_unique_id,
                           self.left_rod_link,
                           computeLinkVelocity=1)[7])
        left_rotate_angular_velocity = left_sum_angular_velocity - left_stroke_angular_velocity

        #getLinkState[5]：URDF worldLinkFrameOrientation，vec4, list of 4 floats
        right_orientation = np.array(
            p.getMatrixFromQuaternion(
                p.getLinkState(self.body_unique_id,
                               self.right_wing_link)[5])).reshape(3, 3)

        left_orientation = np.array(
            p.getMatrixFromQuaternion(
                p.getLinkState(self.body_unique_id,
                               self.left_wing_link)[5])).reshape(3, 3)

        right_r_axis = right_orientation[:, self.params.rotate_axis]
        right_c_axis = -1 * right_orientation[:, self.params.stroke_axis]
        if right_stroke_angular_velocity.dot(right_c_axis) > 0:
            right_z_axis = right_orientation[:, self.params.the_left_axis]
        else:
            right_z_axis = -1 * right_orientation[:, self.params.the_left_axis]

        left_r_axis = -1 * left_orientation[:, self.params.rotate_axis]
        left_c_axis = -1 * left_orientation[:, self.params.stroke_axis]
        if left_stroke_angular_velocity.dot(left_c_axis) < 0:
            left_z_axis = left_orientation[:, self.params.the_left_axis]
        else:
            left_z_axis = -1 * left_orientation[:, self.params.the_left_axis]

        return [right_stroke_angular_velocity,
                right_rotate_angular_velocity,
                right_c_axis, right_r_axis, right_z_axis,
                left_stroke_angular_velocity,
                left_rotate_angular_velocity,
                left_c_axis, left_r_axis, left_z_axis]

    def set_link_force_world_frame(self,
                                   link_id,
                                   position_bias: np.ndarray,
                                   force: np.ndarray):
        """
        """
        force = np.array(force)

        if force.dot(force) == 0:
            return
        #getLinkState[4]：worldLinkFramePosition，vec3, list of 3 floats
        pos = p.getLinkState(self.body_unique_id,
                             link_id)[4]
        pos = np.array(pos)
        position_bias = np.array(position_bias)
        #Use applyExternalForce and applyExternalTorque to apply force or torque to an entity.
        # Please note that this method is only effective when using stepSimulation to explicitly step the simulation.
        p.applyExternalForce(self.body_unique_id,
                             link_id,
                             forceObj=force,
                             posObj=pos + position_bias,
                             flags=p.WORLD_FRAME)
        self.draw_a_line(pos + position_bias,
                         pos + position_bias + force,
                         [1, 0, 0],
                         f'{link_id}')

    def set_link_torque_world_frame(self,
                                    linkid,
                                    torque: np.ndarray):
        """
        """
        torque = np.array(torque)
        if torque.dot(torque) == 0:
            return
        p.applyExternalTorque(self.body_unique_id,
                              linkid,
                              torqueObj=torque,
                              flags=p.WORLD_FRAME)

    def draw_a_line(self,
                    start,
                    end,
                    line_color,
                    name):
        if "debugline" not in self.data:
            self.data["debugline"] = {}
        if name not in self.data["debugline"]:
            self.data["debugline"][name] = -100
        self.data["debugline"][name] = p.addUserDebugLine(start,
                                                          end,
                                                          lineColorRGB=line_color,
                                                          replaceItemUniqueId=self.data["debugline"][name])

    def get_constraint_state(self):
        if self.if_fixed is False:
            #raise AttributeError("the force&torque sensor is not enabled")
            return [0,0,0,0,0,0]
        #getConstraintState：Given a unique constraint id, you can query the applied constraint force in the most recent simulation step. 
        # The input is a unique constraint id, and the output is a constraint force vector, 
        # whose dimension is the degree of freedom affected by the constraint (for example, a fixed constraint affects 6 degrees of freedom).
        return p.getConstraintState(self.constraint_id)

    def reset(self):
        """
        reset the base position orientation and velocity
        """
        self.logger.debug("The MAV is Reset")
        p.resetBasePositionAndOrientation(self.body_unique_id,
                                          posObj=self.params.initial_xyz,
                                          ornObj=self.params.initial_orientation)

    def close(self):
        """
        close this environment
        """
        self.logger.debug("The MAV is closed")
        p.removeBody(self.body_unique_id)
        p.disconnect(self.physics_client)

    def housekeeping(self):
        """
        all the cache will be removed
        """
        self.data.clear()
