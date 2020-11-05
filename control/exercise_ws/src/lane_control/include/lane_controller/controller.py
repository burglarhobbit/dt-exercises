import numpy as np
import traceback

class PurePursuitLaneController:
    """
    The Lane Controller can be used to compute control commands from pose estimations.

    The control commands are in terms of linear and angular velocity (v, omega). The input are errors in the relative
    pose of the Duckiebot in the current lane.
    """

    def __init__(self, parameters):

        self.parameters = parameters

    def update_parameters(self, parameters):
        """Updates parameters of LaneController object.

            Args:
                parameters (:obj:`dict`): dictionary containing the new parameters for LaneController object.
        """
        self.parameters = parameters

    def compute_d(self, x, y):

        return np.sqrt(x**2 + y**2)

    def process_segments(self, segment_msg_list):
        """
            WHITE=0
            uint8 YELLOW=1  
            uint8 RED=2
        """

        WHITE, YELLOW, RED = 0, 1, 2
        d = 0
        min_d = 100000000
        data = []
        follow_point = (0, 0)

        white_pts = []
        yellow_pts = []
        white_dist = []
        yellow_dist = []
        
        yellow_tangents = []
        white_tangents = []
        # print("self.parameters:", self.parameters["~lookahead_dist"].value)
        for seg in segment_msg_list.segments:
            
            x1 = seg.points[0].x
            y1 = seg.points[0].y
            x2 = seg.points[1].x
            y2 = seg.points[1].y

            norm_x = seg.normal.x
            norm_y = seg.normal.y

            mid_x = (x1+x2)/2
            mid_y = (y1+y2)/2

            tangent = -(x2-x1)/(y2-y1) # -ve because flipped co-ord system

            d = self.compute_d(mid_x,mid_y)
            diff = self.parameters["~lookahead_dist"].value - d

            if seg.color == seg.YELLOW:
                yellow_pts.append([mid_x, mid_y])
                yellow_dist.append(d)
                yellow_tangents.append(tangent)
            elif seg.color == seg.WHITE:
                white_pts.append([mid_x, mid_y])
                white_dist.append(d)
                white_tangents.append(tangent)

            # d1 = self.compute_d(x1,y1)
            # d2 = self.compute_d(x2,y2)
            # diff1 = self.parameters["L"] - d1
            # diff2 = self.parameters["L"] - d2
            # if diff1 > 0 and diff1 < min_d:
            #     x,y = x1,y1
            # if diff2 > 0 and diff2 < min_d:
            #     x,y = x2,y2
        print("Yellow points:")
        for i in zip(yellow_pts,yellow_dist,yellow_tangents):
            print(i)
        print("White points:")
        for i in zip(white_pts,white_dist,white_tangents):
            print(i)
        """ 3 cases:
                1. if left/right white points are there: x = (x1+x2)*3/4
                2. if left white point + yellow points are there: x = (x1+x2)*3/2
                3. if right white point + yellow points are there: x = (x1+x2)/2
        """
        # return (0,0), False
        point_scalar = 0.5
        displacement_val = 0.1

        yellow_pts_stat, white_pts_stat = False, False
        if len(yellow_pts):
            yellow_pts_stat = True
            yellow_pts = np.array(yellow_pts)
            yellow_dist = np.array(yellow_dist)
            yellow_tangents = np.array(yellow_tangents)
        if len(white_pts):
            white_pts_stat = True
            white_pts = np.array(white_pts)
            white_dist = np.array(white_dist)
            white_tangents = np.array(white_tangents)

        tangent = np.tan(np.deg2rad(45))
        tangent_60 = np.tan(np.deg2rad(60))
        
        if yellow_pts_stat:
            print("In Yellow")
            y_filter_pts_idx = (np.absolute(yellow_pts) < 0.5).all(axis=1) # logical and of dim-1
            yellow_median_idx = np.argsort(yellow_dist)[len(yellow_dist)//2]
            yellow_median = yellow_pts[yellow_median_idx]

            if np.any(y_filter_pts_idx) > 1:
                yellow_pts_filtered = yellow_pts[y_filter_pts_idx]
                yellow_dist_filtered = yellow_dist[y_filter_pts_idx]
                
                m = np.sort(yellow_tangents)[len(yellow_tangents)//2] # np.mean(yellow_tangents)
                pts = yellow_pts

                # turn right
                if m < tangent and m > 0:
                    follow_point_idx = np.argmin(pts, axis = 0)[1]
                    follow_point = pts[follow_point_idx]
                    follow_point[0] -= 0.1 # between the lane
                    follow_point[1] -= 0.1 # a bit towards the right
                # turn left
                elif m > -tangent and m < 0:
                    follow_point_idx = np.argmax(pts, axis = 0)[1]
                    follow_point = pts[follow_point_idx]
                    follow_point[0] += 0.1
                elif white_pts_stat:
                    # filtering points on the right of yellow lane
                    print("elif white_pts_stat")
                    w_filter_pts_idx = np.logical_and(white_pts[:, 1] < yellow_median[1], white_pts[:, 0] > yellow_median[0])
                    if np.any(w_filter_pts_idx):
                        white_pts_filtered = white_pts[w_filter_pts_idx] 
                        print("white pts filtered:",white_pts_filtered)
                        white_median = np.median(white_pts_filtered, axis=0)

                        follow_point = (yellow_median + white_median) / 2  
                        print("yellow_med, white_med",yellow_median,white_median)
                        # move right
                        if m > tangent and m < tangent_60:
                            print("1")
                            follow_point = (white_median[0], white_median[1] + displacement_val)
                        # move left
                        elif m < -tangent and m > -tangent_60:
                            print("2")
                            follow_point = (white_median[0], white_median[1] + displacement_val)
                        else:
                            follow_point = (0,0)
                        return tuple(follow_point), False
                    else:
                        follow_point = (0,0)
                        return tuple(follow_point), False

            else:
                # turn right
                if yellow_median[1] < 0.1:
                    follow_point = (yellow_median[0], yellow_median[1] - displacement_val)  

                # turn left
                elif yellow_median[1] > 0.2:
                    follow_point = (yellow_median[0], yellow_median[1] + displacement_val)

                # do nothing
                else:
                    follow_point = (0,0)
        
        # turn dynamic
        elif white_pts_stat:
            print("In White")
            if len(yellow_pts):
                pass
            else:
                #usually removes the left lane white lines
                w_filter_pts_idx = (np.absolute(white_pts) < 0.5).all(axis=1) # logical and of dim-1
                white_pts_filtered = white_pts[w_filter_pts_idx] 
                white_dist_filtered = white_dist[w_filter_pts_idx]
                white_tangents_filtered = white_tangents[w_filter_pts_idx]

                m = np.sort(white_tangents_filtered)[len(white_tangents_filtered)//2] # np.mean(white_tangents_filtered)
                pts = white_pts_filtered

                white_median_idx = np.argsort(white_dist_filtered)[len(white_dist_filtered)//2]
                white_median = white_pts_filtered[white_median_idx]

                # turn hard right
                if m < tangent and m > 0:
                    follow_point_idx = np.argmin(pts, axis = 0)[1]
                    follow_point = pts[follow_point_idx]
                    follow_point[0] += 0.1 # between the lane
                    follow_point[1] -= 0.1 # a bit towards the right

                # turn hard left
                elif m > -tangent and m < 0:
                    follow_point_idx = np.argmax(pts, axis = 0)[1]
                    follow_point = pts[follow_point_idx]
                    follow_point[0] -= 0.1 #between the lane
                # move right
                elif m > tangent and m < tangent_60:
                    follow_point = (white_median[0], white_median[1] + displacement_val)
                # move left
                elif m < -tangent and m > -tangent_60:
                    follow_point = (white_median[0], white_median[1] + displacement_val)
                else:
                    follow_point = (0,0)
                return tuple(follow_point), False

                # turn right
                if white_median[1] < -0.15:
                    follow_point = (white_median[0], white_median[1] - 0.1)  

                # turn left
                elif white_median[1] > -0.1:
                    follow_point = (white_median[0], white_median[1] + 0.1)

                # do nothing
                else:
                    follow_point = (0,0)

        # do nothing
        else:
            follow_point = (0,0)

        if not yellow_pts_stat and not white_pts_stat:
            use_prev_omega = True
        else:
            use_prev_omega = False

        x, y = follow_point
        
        return (x, y), use_prev_omega

    def pure_pursuit(self, follow_point, v, K=1.0):
        """
        Input:
            - follow_point: numpy array of follow point [x,y] in robot frame
            - K: controller gain
        Return:
            - v: linear velocity in m/s (float)
            - w: angular velocity in rad/s (float)
        """
        
        # compute distance between robot and follow point
        d = np.sqrt(follow_point[0] ** 2 + follow_point[1] ** 2)
        
        # TODO: compute sin(alpha)
        sin_alpha = 2 * follow_point[1] / d
        
        v = v # we can make it constant or we can make it as a function of sin_alpha
        
        # TODO: compute angular velocity
        w = sin_alpha / K

        # slow down
        if np.abs(w) > 1.0:
            v = 0.1
        
        return v, w
