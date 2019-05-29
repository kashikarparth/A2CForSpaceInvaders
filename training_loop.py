import tensorflow as tf
tf.reset_default_graph()

import numpy as np
import gym
from skimage import transform
from skimage.color import rgb2gray
from collections import deque
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
import time 

###########################################################################################################################
def conv_layer(inputs, filters, kernel_size, strides, gain=1.0):
    return tf.layers.conv2d(inputs=inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(strides, strides),
                            activation=tf.nn.relu,
                            padding = "VALID",
                            kernel_initializer =tf.orthogonal_initializer(gain=gain))

# Fully connected layer
def fc_layer(inputs, units, activation_fn=tf.nn.relu, gain=1.0):
    return tf.layers.dense(inputs=inputs,
                           units=units,
                           activation=activation_fn,
                           kernel_initializer=tf.orthogonal_initializer(gain))


def a2cnet(state, isTrain = True, reuse = False):
    with tf.variable_scope('a2cnetwork', reuse = reuse):
        gain = np.sqrt(2)
        conv1 = conv_layer(state, 32, 8, 4, gain)
        conv2 = conv_layer(conv1, 64, 4, 2, gain) 
        conv3 = conv_layer(conv2, 64, 3, 2, gain)
        flatten1 = tf.layers.flatten(conv3)
        fc_common = fc_layer(flatten1, 512, gain = gain)
        
        fc_probability_layer = tf.nn.softmax(fc_layer(fc_common, 6))
        
        fc_value_layer = fc_layer(fc_common, 1, activation_fn = None)
        
        return fc_probability_layer, fc_value_layer

stacked_batch = tf.placeholder(dtype = tf.float32, shape = [None, 110,84,4])
reward_vals = tf.placeholder(dtype = tf.float32, shape = [None,1])
actions_taken = tf.placeholder(dtype = tf.int32, shape = [None,2])
observation_stack = tf.placeholder(dtype = tf.float32, shape = [None,110,84,4])


inference = a2cnet(observation_stack)

action_probabilities, state_values = a2cnet(stacked_batch, reuse = True)
advantage_values = tf.subtract(reward_vals,state_values)
value_loss = tf.reduce_mean(tf.square(advantage_values))

log_action_probabilities = tf.log(tf.gather_nd(action_probabilities,actions_taken))
policy_loss =tf.reduce_mean(-1 * tf.multiply(log_action_probabilities,advantage_values))
total_loss = value_loss + policy_loss

lr = 0.001
optimizer = tf.train.AdamOptimizer(lr)
gradients, variables = zip(*optimizer.compute_gradients(total_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
optimize = optimizer.apply_gradients(zip(gradients, variables))

#######################################################################################################################3

env = gym.make('SpaceInvaders-v0')
possible_actions = np.array(np.identity(env.action_space.n,dtype=int).tolist())

def preprocess_frame(frame):
    gray = rgb2gray(frame)
    cropped_frame = gray[8:-12,4:-12]
    normalized_frame = cropped_frame/255.0
    preprocessed_frame = transform.resize(normalized_frame,[110,84])
    return preprocessed_frame


stack_size = 4
stacked_frames = deque([np.zeros((110,84),dtype=np.int) for i in range(stack_size)],maxlen = 4)

def stack_frames(stacked_frames, state, is_new_episode = False):
    frame = preprocess_frame(state)
    
    if is_new_episode:
        stacked_frames = deque([np.zeros((110,84),dtype=np.int) for i in range(stack_size)],maxlen = 4)
        
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        
        stacked_state = np.stack(stacked_frames,axis=2)
    
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames,axis=2)
    
    return stacked_state, stacked_frames
###################################################################################################################
#TRAINING LOOP
    
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()



def training_loop_per_episode(max_steps_per_episode, sess, env, gamma = 0.9):
    R = 0
    stacked_frames = deque([np.zeros((110,84),dtype=np.int) for i in range(stack_size)],maxlen = 4)

    #Placeholder_target_values
    actions = []
    r_i = []
    states_stack = []
    
    #Iniial_stack_tensor_ready
    observation = env.reset()
    observation_preprocessed = preprocess_frame(observation)
    observation_stacked,stacked_frames = stack_frames(stacked_frames, observation_preprocessed, True)
    observation_stacked_rs = np.reshape(observation_stacked, newshape = [-1,110,84,4])

    for i_step in range(max_steps_per_episode):
        var = np.random.uniform()
        
        if var>=0.2:
            action_taken,v_s = sess.run(inference,{observation_stack: observation_stacked_rs})
            action_taken = np.argmax(possible_actions[np.argmax(action_taken)])
            
        else:
            _,v_s = sess.run(inference,{observation_stack: observation_stacked_rs})
            action_taken = np.random.randint(low = 0, high = 6)
        
        observation_next,reward,done,info = env.step(action_taken)
        observation_next_preprocessed = preprocess_frame(observation_next)
        
        states_stack.append(observation_stacked)
        actions.append((i_step,action_taken))
        r_i.append(reward)
        
        if not done:
            R = v_s[0][0]
            observation = observation_next_preprocessed
            
            observation_stacked,stacked_frames = stack_frames(stacked_frames, observation_preprocessed, False)
            observation_stacked_rs = np.reshape(observation_stacked, newshape = [-1,110,84,4])
        
        if done:
            R = 0
            break

    for r_t in range(len(r_i)-1, -1, -1):
        R = r_i[r_t] + gamma*R
        r_i[r_t] = R

    return states_stack, actions, r_i
         

def training_loop(n_episodes, max_steps_per_episode, sess, env, gamma = 0.9):
    lowest_loss = 100
    loss_list = []
    print("-----------------------Training started! ---------------------------------------")
    for episode in range(n_episodes):
        t = time.time()
        states_stack_, actions_, R_ = training_loop_per_episode(max_steps_per_episode, sess, env, gamma = 0.9)
        time_taken = time.time()-t
        R_ = np.reshape(R_, newshape = [-1,1])
        _,totloss = sess.run([optimize, total_loss],{stacked_batch: states_stack_,actions_taken:actions_,reward_vals: R_})
        print("Total loss for episode " +str(episode+1)+ " is " + str(totloss) + " in " + str(time_taken) + " seconds.")
        loss_list.append(totloss)
        if totloss<lowest_loss:
            lowest_loss = totloss
            saver.save(sess, "/home/kashikar/Drive/AI/Projects/4x4TicTacToe/Remake/model.ckpt")
    plt.plot(range(n_episodes),loss_list)

training_loop(2000,1000,sess,env)