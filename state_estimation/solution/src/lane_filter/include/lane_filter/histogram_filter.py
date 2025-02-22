#!/usr/bin/env python
# coding: utf-8

# In[19]:


# start by importing some things we will need
import cv2
import matplotlib
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import entropy, multivariate_normal
from math import floor, sqrt

# In[22]:


# Now let's define the prior function. In this case we choose
# to initialize the historgram based on a Gaussian distribution
def histogram_prior(belief, grid_spec, mean_0, cov_0):
    pos = np.empty(belief.shape + (2,))
    pos[:, :, 0] = grid_spec["d"]
    pos[:, :, 1] = grid_spec["phi"]
    RV = multivariate_normal(mean_0,cov_0)
    belief = RV.pdf(pos)
    return belief

# In[111]:


# Now let's define the predict function


def histogram_predict(belief, dt, left_encoder_ticks, right_encoder_ticks, grid_spec, robot_spec, cov_mask):
        belief_in = belief
        delta_t = dt
        
        # TODO calculate v and w from ticks using kinematics. You will need `robot_spec`
        v = 0.0 # replace this with a function that uses the encoder 
        w = 0.0 # replace this with a function that uses the encoder
        
        R = robot_spec['wheel_radius']
        L = robot_spec['wheel_baseline']
        alpha = (2*np.pi)/robot_spec['encoder_resolution']
        
        
        left_wheel_move = alpha * left_encoder_ticks * R
        right_wheel_move = alpha * right_encoder_ticks * R
        
        dist_travel = (left_wheel_move + right_wheel_move)/2
        
        v = dist_travel
        w = (right_wheel_move - left_wheel_move)/(L)
        
        #print((v*delta_t/grid_spec['delta_d']))
        #print(w)
        #input()
        
        # TODO propagate each centroid forward using the kinematic function
        
        d_t = grid_spec['d']  + v*np.sin(w)# replace this with something that adds the new odometry
        phi_t = grid_spec['phi'] + w# replace this with something that adds the new odometry
        
        # replace this with something that adds the new odometry
        
        p_belief = np.zeros(belief.shape)

        # Accumulate the mass for each cell as a result of the propagation step
        for i in range(belief.shape[0]):
            for j in range(belief.shape[1]):
                # If belief[i,j] there was no mass to move in the first place
                if belief[i, j] > 0:
                    # Now check that the centroid of the cell wasn't propagated out of the allowable range
                    if (
                        d_t[i, j] > grid_spec['d_max']
                        or d_t[i, j] < grid_spec['d_min']
                        or phi_t[i, j] < grid_spec['phi_min']
                        or phi_t[i, j] > grid_spec['phi_max']
                    ):
                        
                        continue
                    
                    # TODO Now find the cell where the new mass should be added
                    
                    i_new =  i + int(np.floor( (v*np.sin(w)) /grid_spec['delta_d'])) # replace with something that accounts for the movement of the ro
                    j_new = j+int(np.floor((w) /grid_spec['delta_phi'])) # replace with something that accounts for the movement of the robot
                    #i_new = i
                    #j_new = j
                    if j_new < p_belief.shape[1] and i_new < p_belief.shape[0] :
                        p_belief[i_new, j_new] += belief[i, j]
                        

        # Finally we are going to add some "noise" according to the process model noise
        # This is implemented as a Gaussian blur
        s_belief = np.zeros(belief.shape)
        gaussian_filter(p_belief, cov_mask, output=s_belief, mode="constant")
        
        if np.sum(s_belief) == 0:
            return belief_in
        belief = s_belief / np.sum(s_belief)
        return belief


# In[112]:


# We will start by doing a little bit of processing on the segments to remove anything that is behing the robot (why would it be behind?)
# or a color not equal to yellow or white

def prepare_segments(segments):
    filtered_segments = []
    for segment in segments:

        # we don't care about RED ones for now
        if segment.color != segment.WHITE and segment.color != segment.YELLOW:
            continue
        # filter out any segments that are behind us
        if segment.points[0].x < 0 or segment.points[1].x < 0:
            continue

        filtered_segments.append(segment)
    return filtered_segments

# In[113]:



def generate_vote(segment, road_spec):
    p1 = np.array([segment.points[0].x, segment.points[0].y])
    p2 = np.array([segment.points[1].x, segment.points[1].y])
    t_hat = (p2 - p1) / np.linalg.norm(p2 - p1)
    n_hat = np.array([-t_hat[1], t_hat[0]])
    
    d1 = np.inner(n_hat, p1)
    d2 = np.inner(n_hat, p2)
    l1 = np.inner(t_hat, p1)
    l2 = np.inner(t_hat, p2)
    if l1 < 0:
        l1 = -l1
    if l2 < 0:
        l2 = -l2

    l_i = (l1 + l2) / 2
    d_i = (d1 + d2) / 2
    phi_i = np.arcsin(t_hat[1])
    if segment.color == segment.WHITE:  # right lane is white
        if p1[0] > p2[0]:  # right edge of white lane
            d_i -= road_spec['linewidth_white']
        else:  # left edge of white lane
            d_i = -d_i
            phi_i = -phi_i
        d_i -= road_spec['lanewidth'] / 2

    elif segment.color == segment.YELLOW:  # left lane is yellow
        if p2[0] > p1[0]:  # left edge of yellow lane
            d_i -= road_spec['linewidth_yellow']
            phi_i = -phi_i
        else:  # right edge of white lane
            d_i = -d_i
        d_i = road_spec['lanewidth'] / 2 - d_i

    return d_i, phi_i

# In[114]:


def generate_measurement_likelihood(segments, road_spec, grid_spec):

    # initialize measurement likelihood to all zeros
    measurement_likelihood = np.zeros(grid_spec['d'].shape)#TODO change it back to zeros

    for segment in segments:
        d_i, phi_i = generate_vote(segment, road_spec)

        # if the vote lands outside of the histogram discard it
        if d_i > grid_spec['d_max'] or d_i < grid_spec['d_min'] or phi_i < grid_spec['phi_min'] or phi_i > grid_spec['phi_max']:
            continue

        # TODO find the cell index that corresponds to the measurement d_i, phi_i
        i = int( np.floor((d_i - grid_spec['d_min'])/grid_spec['delta_d']) )# replace this
        j = int ( np.floor((phi_i - grid_spec['phi_min'])/grid_spec['delta_phi']) )# replace this
        
        # Add one vote to that cell      
        if i < measurement_likelihood.shape[0] and j < measurement_likelihood.shape[1]:
            measurement_likelihood[i, j] += 1

    if np.linalg.norm(measurement_likelihood) == 0:
        return None
    measurement_likelihood /= np.sum(measurement_likelihood)
    return measurement_likelihood


# In[115]:


def histogram_update(belief, segments, road_spec, grid_spec):
    # prepare the segments for each belief array
    segmentsArray = prepare_segments(segments)
    # generate all belief arrays

    measurement_likelihood = generate_measurement_likelihood(segmentsArray, road_spec, grid_spec)

    ones_1 = np.ones(belief.shape)
        
    ones_1 /= np.sum(ones_1)
    
    
    if measurement_likelihood is not None:
        # TODO: combine the prior belief and the measurement likelihood to get the posterior belief
        # Don't forget that you may need to normalize to ensure that the output is valid probability distribution
        #print(belief)
        #print(belief.shape)
        #input()
   
        
#         Z = 500./float(ones_1.shape[0]*ones_1.shape[1])
#         measurement_likelihood = gaussian_filter( (measurement_likelihood), sigma=0.1)
#         scale = 1
#         measurement_likelihood /= np.sum(measurement_likelihood)
       
         
        #x, y = np.meshgrid(np.linspace(-1, 1, kernel_size),np.linspace(-1, 1, kernel_size))\
        #sigma = 12.0
        #d, phi = np.mgrid[grid_spec['d_min'] : grid_spec['d_max'] : grid_spec['delta_d'], grid_spec['phi_min'] : grid_spec['phi_max'] : grid_spec['delta_phi']]
        #dst = np.sqrt(d**2+phi**2)
    # lower normal part of gaussian
        #normal = 1/(2.0 * np.pi * sigma**2)
    # Calculating Gaussian filter
        #gauss = np.exp(-((dst)**2 / (2.0 * sigma**2))) * normal
        
        belief = np.multiply(belief, measurement_likelihood) # replace this with something that combines the belief and the measurement_likelihood    
        #belief = np.multiply(belief, gauss)
        
        if np.sum(belief) == 0:
            #RV = multivariate_normal([0,0],[[0.01,0],[0,0.01]])
            #belief = RV.pdf(pos)
            belief = measurement_likelihood
          
        belief = belief / np.sum(belief)
    
        
    return (measurement_likelihood, belief)

