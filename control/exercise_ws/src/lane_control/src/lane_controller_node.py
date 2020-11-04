#!/usr/bin/env python3
import numpy as np
import rospy

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, WheelsCmdStamped, BoolStamped, FSMState, StopLineReading, SegmentList

from lane_controller.controller import PurePursuitLaneController


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocitie.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )

        self.omega_repeat_count = 0
        self.v = 0.4
        self.v_prev = 0.4
        self.omega_prev = 0.0

        # Add the node parameters to the parameters dictionary
        self.params = dict()

        # lookahead distance
        self.params['~lookahead_dist'] = DTParam(
            '~lookahead_dist',
            param_type=ParamType.FLOAT,
            min_value=0.0,
            max_value=10.0
        )

        self.pp_controller = PurePursuitLaneController(self.params)

        # Construct publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd",
                                           Twist2DStamped,
                                           queue_size=1,
                                           dt_topic_type=TopicType.CONTROL)

        # Construct subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose",
                                                 LanePose,
                                                 self.cbLanePoses,
                                                 queue_size=1)

        # self.sub_segment_list = rospy.Subscriber("/agent/line_detector_node/segment_list",
        #                            SegmentList,
        #                            self.cbSegmentLists,
        #                            queue_size=1)

        self.sub_filtered_segment_list = rospy.Subscriber("/agent/lane_filter_node/seglist_filtered",
                                   SegmentList,
                                   self.cbFilteredSegmentLists,
                                   queue_size=1)

        self.log("Initialized!")

    def cbLanePoses(self, input_pose_msg):
        """Callback receiving pose messages

        Args:
            input_pose_msg (:obj:`LanePose`): Message containing information about the current lane pose.
        """
        self.pose_msg = input_pose_msg
        # print("self.pose_msg:", self.pose_msg)

        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        # TODO This needs to get changed
        car_control_msg.v = self.v
        car_control_msg.omega = 0

        self.publishCmd(car_control_msg)

    def cbSegmentLists(self, input_seg_msg):
        pass

    def cbFilteredSegmentLists(self, input_seg_msg):
        """Callback receiving a list of the detected segments
        https://github.com/duckietown/dt-core/blob/daffy/packages/line_detector/src/line_detector_node.py
        
        Args:
            input_seg_msg (:obj:`SegmentList`): Message containing information about the detected segments
        """
        self.seg_msg = input_seg_msg
        # print("self.seg_msg:", self.seg_msg)
    
        car_control_msg = Twist2DStamped()
        car_control_msg.header = self.pose_msg.header

        if self.omega_repeat_count:
            self.omega_repeat_count -= 1
            car_control_msg.v = self.v_prev
            car_control_msg.omega = self.omega_prev
            print("Setting v, omega value to:", self.v_prev, self.omega_prev)
            self.publishCmd(car_control_msg)
            return

        follow_point, use_prev_omega = self.pp_controller.process_segments(self.seg_msg)
        print("Follow point:", follow_point)
        if follow_point != (0,0):
            v, w = self.pp_controller.pure_pursuit(follow_point, self.v)
            self.log("Omega: %.2f"%w)

            car_control_msg.v = v
            car_control_msg.omega = w
            print("Setting v, omega value to: (%0.2f, %.2f)" % (self.v, self.omega_prev))
            self.omega_prev = w
            self.v_prev = v
            if np.abs(w) > 1:
                self.omega_repeat_count = 8
        else:
            # TODO This needs to get changed
            car_control_msg.v = self.v
            if use_prev_omega:
                car_control_msg.omega = self.omega_prev
            else:
                car_control_msg.omega = 0

        self.publishCmd(car_control_msg)

    def publishCmd(self, car_cmd_msg):
        """Publishes a car commsand message.

        Args:
            car_cmd_msg (:obj:`Twist2DStamped`): Message containing the requested control action.
        """
        self.pub_car_cmd.publish(car_cmd_msg)


    def cbParametersChanged(self):
        """Updates parameters in the controller object."""

        self.controller.update_parameters(self.params)


if __name__ == "__main__":
    # Initialize the node
    lane_controller_node = LaneControllerNode(node_name='lane_controller_node')
    # Keep it spinning
    rospy.spin()
