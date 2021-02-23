import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from celluloid import Camera



def GridGen(x_range,y_range,vertex, plot):
    
    #generate the target space with dimensions approximately tgrid_x,tgrid_y
    #one moderately tricky thing is we want 0,0 to be in the grid
    
    xs = np.arange(start=-x_range-.5*3**.5,stop=x_range,step=vertex*3**.5) #beginning x array
    if len(xs) % 2 == 0: #check number of columns
        xs = xs[1:len(xs)] #coerce to an odd number of columns
    xs = xs-np.median(xs) # we want origin at actual 0,0
    
    ys = np.arange(start=-y_range,stop=y_range,step=vertex)
    if len(ys) % 2 == 0: #check number of rows (half the number actually)
        ys = ys[1:len(ys)] #coerce to an odd number of columns
    ys = ys-np.median(ys) # we want origin at actual 0,0
    
    targs = np.zeros((2,len(xs))) #initialize the targets array
    
    for i in ys:
    
        t_int1 = np.vstack((xs,np.full((1,len(xs)),i))) #a single row of "major axis" targets

        t_int2 = np.vstack((xs+(vertex/2)*3**.5,np.full((1,len(xs)),i+vertex/2))) #a single row of "minor axis" targets

        targs = np.hstack((targs,t_int1))
        targs = np.hstack((targs,t_int2))
    
    Grid = np.round(targs[:,len(xs):np.shape(targs)[1]],3).T
    num_targets = Grid.shape[0]
    
    Grid_dist = np.zeros((num_targets,num_targets)) # Matrix containing pairwise target distanes

    for i in range(num_targets): # Calculate the euclidean distance for each pair of targets
        for j in range(num_targets):
            Grid_dist[i,j] = np.linalg.norm(Grid[i, :]-Grid[j, :])
            
    # Plots the grid + the center target and its neighbors
    if plot:
        which = np.where(np.logical_and(Grid[:,0]==0 , Grid[:,1]==0)==1)[0][0]
        neighbor_idx = np.where(np.round(Grid_dist[which,:]) == vertex)[0]
        plt.scatter(Grid[:,0],Grid[:, 1])
        plt.scatter(Grid[which, 0],Grid[which, 1], color='C3')
        plt.scatter(Grid[neighbor_idx, 0],Grid[neighbor_idx, 1], color='C5')
        plt.show()
    
    return Grid, Grid_dist



# Sequence with Jump
# jump type:  [0] No Jump,   [1] Only +2   [2] Both +1 and +2
def SeqGen(Grid, Grid_dist, seq_length, vertex, jump, plot):
    status = True # Final status of current trial. Is it a successful trial?
    num_targets = Grid.shape[0]
    Trajectory = np.zeros((seq_length,), dtype=np.int)
    Trajectory[0] = np.where(np.logical_and(Grid[:,0]==0 , Grid[:,1]==0)==1)[0]
    epsi = np.finfo(float).eps # Used for neumrical stability of ArcCos
    
    Jump = np.zeros((seq_length,3), dtype=np.int) # Empty list for saving jump information
    jump_range = np.arange(4, seq_length - 5)    # Where in sequence jump happen, neither in first 5 nor in last 5
    jump_inx = np.random.permutation(jump_range)[0] # Randomly select one target from jump_range to put the jump
    
    for i in range(seq_length-1):
        if i == 0: # For the first reach
            # Find the neighbors for current possition: Possible targets 
            neighbor_idx = np.where(np.round(Grid_dist[Trajectory[i],:]) == vertex)[0]
            next_idx = np.random.permutation(neighbor_idx)[0]
            Trajectory[i+1] = next_idx
        else:
            # Find the neighbors for current possition: Possible targets 
            neighbor_idx = np.where(np.round(Grid_dist[Trajectory[i],:]) == vertex)[0]


            # Remove the last position from the list
            neighbor_idx = np.delete(neighbor_idx,np.where(neighbor_idx==Trajectory[i-1]))
            neighbor_idx = np.delete(neighbor_idx,np.where(neighbor_idx==Trajectory[i-2]))
            
            # Regulating reaches
            a = Grid[Trajectory[i],:] - Grid[Trajectory[i-1],:] # the vector for previous move

            # Augmented possible neighbors after probability correction
            neighbor_idx_aug = neighbor_idx
            for idx in neighbor_idx:
                # The vector for all possible next moves
                b = Grid[idx,:] - Grid[Trajectory[i],:]
                # Angle of possible next movement
                ang = np.round(np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)) - epsi)*(180/np.pi))
                # increace the probability if the possible move is 60 degrees
                if ang == 60:
                    neighbor_idx_aug = np.append(neighbor_idx_aug, idx)
                    neighbor_idx_aug = np.append(neighbor_idx_aug, idx)
     
            if len(neighbor_idx_aug) == 0:
                print('Ran out of choices')
                status = False
                continue
            
            next_idx = np.random.permutation(neighbor_idx_aug)[0]
            Trajectory[i+1] = next_idx
            
            
            if i == jump_inx+1:
                if jump == 2:
                    Jump[i-1, 0] = 2
                    hoax_next_idx = np.random.permutation(neighbor_idx_aug[neighbor_idx_aug!=next_idx])[0] # Remove the main choice form the list of possible jumps and select and alternative randomly
                    Jump[i-1, 1] = hoax_next_idx
                    
                    ### And possible targets based on the +1 hoax
                    # Find the neighbors for current possition: Possible targets 
                    neighbor_idx = np.where(np.round(Grid_dist[hoax_next_idx,:]) == vertex)[0]
                    # Remove the last position from the list
                    neighbor_idx = np.delete(neighbor_idx,np.where(neighbor_idx==next_idx))
                    # Regulating reaches
                    a = Grid[hoax_next_idx,:] - Grid[next_idx,:] # the vector for previous move
                    # Augmented possible neighbors after probability correction
                    neighbor_idx_aug = neighbor_idx
                    for idx in neighbor_idx:
                        # The vector for all possible next moves
                        b = Grid[idx,:] - Grid[hoax_next_idx,:]
                        # Angle of possible next movement
                        ang = np.round(np.arccos(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)) - epsi)*(180/np.pi))
                        # increace the probability if the possible move is 60 degrees
                        if ang == 60:
                            neighbor_idx_aug = np.append(neighbor_idx_aug, idx)


                    next_next_hoax = np.random.permutation(neighbor_idx_aug)[0]
                    
                    Jump[i-1, 2] = next_next_hoax
                    
        
            if i == jump_inx+2:
                if jump == 1:
                    Jump[i-2, 0] = 1
                    Jump[i-2, 1] = -1
                    
                    # Check if there are any possible future target choices
                    if len(neighbor_idx_aug[neighbor_idx_aug!=next_idx]) == 0:
                        print("Ran out of possible choices for jump")
                        status = False
                        continue
                        
                    hoax_next_idx = np.random.permutation(neighbor_idx_aug[neighbor_idx_aug!=next_idx])[0] # Remove the main choice form the list of possible jumps and select and alternative randomly
                    Jump[i-2, 2] = hoax_next_idx

        
    return Trajectory, Jump, status


def SeqAnimate(Grid, Trajectory,Jump, name_idx):
    edg = 1
    fig, ax = plt.subplots()
    ax = plt.axes(xlim=(Grid[:,0].min()-edg, Grid[:,0].max()+edg), ylim=(Grid[:, 1].min()-edg, Grid[:, 1].max()+edg))

    camera = Camera(fig)
    x = []
    y = []
    for i in range(len(Trajectory)-1):

        x.append([Grid[Trajectory[i],0] ,Grid[Trajectory[i+1],0]])
        y.append([Grid[Trajectory[i],1] ,Grid[Trajectory[i+1],1]])

        ## +0 case
        ax.scatter(Grid[:,0],Grid[:, 1], color='C0', alpha=0.2)
        if Jump[i,0]!=0:
            ax.scatter(Grid[Jump[i,1],0],Grid[Jump[i,1],1], color='C7')
            ax.scatter(Grid[Jump[i,2],0],Grid[Jump[i,2],1], color='C7')
        
        ax.scatter(Grid[Trajectory[i+1],0],Grid[Trajectory[i+1],1], color='C3')
        ax.plot(x, y, color='C0')
        camera.snap()
        
    animation = camera.animate()  
    animation.save('./Animations/Seq_'+str(name_idx)+'.gif', writer = 'imagemagick')
    
def TPGen(Grid, SEQ, JUMP):
    # Trial Parameters
    Jump_type = 0        # A Default Value, will chance for each trial.
    Jump_which_reach = 0 # A Default Value, will chance for each trial.
    Jump_target_1 = -1   # A default value, will chance for each trial.
    Jump_target_2 = -1   # A default value, will chance for each trial.
    Jump_distance = 0.03    # Distance from the +0 target for showing the jump 
    Start_target_delay = 500
    Between_target_dwell_time = 200
    Max_reach_time = 5000
    Post_trial_delay = 0 
    horizen = 2 # A default value, will chance for each trial.
    current_target_opacity = 100
    N1_opacity = 75 
    N2_opacity = 50 
    N3_opacity = 25 
    N4_opaciy =  5
    
    # Grid Parameters 
    
    
    TP_Trial = np.zeros((SEQ.shape[0], 30))
    # TP_Grid = np.zerso((SEQ.shape[0], )) # Maybe for later, when the grid has also multiple parameters 
    
    for i in range(SEQ.shape[0]):
        TP_Trial[i,0:15] = SEQ[i,:] + 1 # For matlab compatibility
        
        if any(JUMP[i,:,0]!=0):  # In case there is a jump in the trial
            trial_to_jump = np.where(JUMP[i,:,0]!=0)[0]
            TP_Trial[i,15:20] = [JUMP[i, trial_to_jump, 0], trial_to_jump +1, JUMP[i, trial_to_jump, 2] + 1 , JUMP[i, trial_to_jump, 1] + 1, Jump_distance]  # +1 is for matlab compatibility
        else:
            TP_Trial[i,15:20] = [Jump_type, Jump_which_reach, Jump_target_2 + 1, Jump_target_1 +1 , Jump_distance]  # +1 is for matlab compatibility
        
        TP_Trial[i,20:] = [Start_target_delay, Between_target_dwell_time, Max_reach_time, Post_trial_delay, horizen, current_target_opacity, N1_opacity, N2_opacity, N3_opacity, N4_opaciy]
        
    np.savetxt('./TP_Trial.txt', TP_Trial ,fmt='%1.4f', delimiter='\t') # +1 is for matlab compatibility
    print('Wrote Sequences.')
    np.savetxt('./Grid.txt',Grid,fmt='%1.2f', delimiter ='\t')
    print('Wrote Grid Locations.')