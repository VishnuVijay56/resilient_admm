# Full library imports
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Function imports
from tqdm import tqdm
from copy import deepcopy



### Initializations

x_dim = 3
y_dim = 3
iterations = 500
np.random.seed(6050)
add_attack = True
attack_center = 0
attack_scale = 1
adaptive_penalty = True
show_byz_plt = False

show_prob1 = False
show_prob2 = False

# Honest and Rogue Agents
num_agents = 4
num_byz = 1
ind_byz = 2

# Neighor List - Agents are connected to each other
nbr_list = [None] * num_agents
arcs = []
for i in range(num_agents):
    nbr_list_i = []
    for j in range(num_agents):
        if i != j:
            nbr_list_i.append(j)
            arcs.append([i, j])
    nbr_list[i] = nbr_list_i

# Quadratic Objective Functions
# f_i(x) = 0.5 || y_i - M_i*x ||^2
y_arr = [None] * num_agents
M_arr = [None] * num_agents

for i in range(num_agents):
    y_i = np.random.normal(0, 1, (y_dim, 1))
    M_i = np.random.normal(0, 1, (y_dim, x_dim))
    
    y_arr[i] = y_i
    M_arr[i] = M_i



### Centralized: Finding optimal for all agents

x_cp = cp.Variable((x_dim, 1))
obj_func = 0
for i in range(num_agents):
    obj_i = 0.5 * cp.power(cp.norm(y_arr[i] - M_arr[i] @ x_cp), 2)
    obj_func += obj_i
prob_all = cp.Problem(cp.Minimize(obj_func))

prob_all.solve()
x_opt_all = np.array(x_cp.value).reshape((-1, 1))
print("\nOptimal x for optimization over all agents: \n\t", x_opt_all.flatten())



### Centralized: Finding optimal for honest agents

x_cp = cp.Variable((x_dim, 1))
obj_func = 0
for i in range(num_agents):
    if i == ind_byz:
        continue
    obj_i = 0.5 * cp.power(cp.norm(y_arr[i] - M_arr[i] @ x_cp), 2)
    obj_func += obj_i
prob_honest = cp.Problem(cp.Minimize(obj_func))

prob_honest.solve()
x_opt_honest = np.array(x_cp.value).reshape((-1, 1))
print("\nOptimal x for optimization over honest agents: \n\t", x_opt_honest.flatten())



### Decentralized: Finding optimal for all agents with ADMM

# Define x-variable
x_cp_arr = []
x_arr = []
for i in range(num_agents):
    x_cp_arr.append(cp.Variable((x_dim, 1)))
    x_arr.append(np.zeros((x_dim, 1)))

# Define local variables for edges
lam_arr = [None] * num_agents
tot_div_arr = [None] * num_agents
z_cp_arr = [None] * num_agents
z_arr = [None] * num_agents

for i in range(num_agents):
    lam_arr_i = {}
    tot_div_arr_i = {}
    z_cp_arr_i = {}
    z_arr_i = {}
    
    for j in nbr_list[i]:
        lam_arr_i[j] = np.zeros((x_dim, 1))
        tot_div_arr_i[j] = np.zeros((x_dim, 1))
        z_cp_arr_i[j] = cp.Variable((x_dim, 1))
        z_arr_i[j] = np.zeros((x_dim, 1))
    
    lam_arr[i] = lam_arr_i
    tot_div_arr[i] = tot_div_arr_i
    z_cp_arr[i] = z_cp_arr_i
    z_arr[i] = z_arr_i
    
# Storage Variables
x_norm_diff_history = [np.zeros(iterations) for i in range(num_agents)]
trust_history = [np.zeros((num_agents, iterations)) for i in range(num_agents)]
lam_norm_history = [np.zeros((num_agents, iterations)) for i in range(num_agents)]

# ADMM Iterations

trust_parameter_scale = 50
trust_parameter_offset = 50
def trust_parameter(param):
    trust = trust_parameter_scale / ((np.linalg.norm(param)) + trust_parameter_offset)
    # trust = 1 - np.linalg.norm(param) / trust_parameter_scale
    # trust = trust_parameter_scale * (np.exp(+np.linalg.norm(param)) + 1, -1)
    return trust

for iter in tqdm(range(iterations), desc="Iterations", leave=False):
    # Store Variables
    for ind in range(num_agents):
        for nbr_ind in nbr_list[i]:
            if ind == nbr_ind:
                continue
            lam_norm_history[ind][nbr_ind, iter] = np.linalg.norm(lam_arr[ind][nbr_ind])
    
    this_x_arr = deepcopy(x_arr)
    
    # x-Update
    for ind in range(num_agents):
        obj_i = 0.5 * cp.power(cp.norm(y_arr[ind] - M_arr[ind] @ x_cp_arr[ind]), 2)
        
        for nbr_ind in nbr_list[ind]:
            aij = 1
            if adaptive_penalty and (ind != ind_byz):
                aij = trust_parameter(tot_div_arr[ind][nbr_ind])
            
            obj_i += x_cp_arr[ind].T @ (lam_arr[ind][nbr_ind] - aij*(this_x_arr[ind] + this_x_arr[nbr_ind]))
            obj_i += aij * cp.power(cp.norm(x_cp_arr[ind]), 2)
        
        # Solve
        prob1 = cp.Problem(cp.Minimize(obj_i), [])
        try:
            prob1.solve(verbose=show_prob1)
        except cp.SolverError:
            print("Solver Error @ Prob 1")
            quit()
        
        # Byzantine Agent Conditional
        new_x = deepcopy(np.array(x_cp_arr[ind].value).reshape((-1, 1)))
        if add_attack and (ind == ind_byz):
            # new_x += attack_center
            new_x += np.random.uniform(attack_center - attack_scale, attack_center + attack_scale)
            # new_x += np.random.normal(attack_center, attack_scale)
            
        # Store
        x_arr[ind] = new_x
        x_norm_diff = np.linalg.norm(new_x - x_opt_all)
        if add_attack:
            x_norm_diff = np.linalg.norm(new_x - x_opt_honest)
        x_norm_diff_history[ind][iter] = x_norm_diff


    # lam-Update
    for ind in range(num_agents):
        
        for nbr_ind in nbr_list[ind]:
            aij = 1
            if adaptive_penalty and (ind != ind_byz):
                aij = trust_parameter(tot_div_arr[ind][nbr_ind]) # lam_arr[ind][nbr_ind]
            trust_history[ind][nbr_ind, iter] = aij
                
            constr_dev = (x_arr[ind] - x_arr[nbr_ind])
            lam_arr[ind][nbr_ind] += (aij*constr_dev)
            tot_div_arr[ind][nbr_ind] += np.abs(aij*constr_dev)
            


# Print ADMM Results
print("\nADMM: local x at each agent")
for i in range(num_agents):     
    print(f"\tAgent {i}: {x_arr[i].flatten()}")

print("\nADMM: local norm(lam) for each agent")
for i in range(num_agents):
    norm_lam_i = []
    for j in range(num_agents):
        if i == j:
            norm_lam_i.append(0)
        else: 
            norm_lam_i.append(np.linalg.norm(lam_arr[i][j]))
    print(f"\tAgent {i}: {norm_lam_i}")
    
print("\nADMM: local trust param for each agent")
for i in range(num_agents):
    norm_tot_div_i = []
    for j in range(num_agents):
        if i == j:
            norm_tot_div_i.append(0)
        else: 
            norm_tot_div_i.append(trust_parameter(tot_div_arr[i][j]))
    print(f"\tAgent {i}: {norm_tot_div_i}")


###     PLOT: Convergence Results

conv_plt = plt.figure()
conv_ax = conv_plt.add_subplot()
for i in range(num_agents):
    
    if (i == ind_byz) and show_byz_plt:
        color = 'orangered'
        conv_ax.plot(np.arange(iterations), x_norm_diff_history[i], label=f"Agent {i+1}", color=color)
        
    elif (i != ind_byz): 
        color = 'darkgray'
        conv_ax.plot(np.arange(iterations), x_norm_diff_history[i], label=f"Agent {i+1}", color=color)

conv_title1 = "Adaptive Penalty: False"
if adaptive_penalty:
    conv_title1 = "Adaptive Penalty: True"
    
conv_title2 = "Attack: False"
if add_attack:
    conv_title2 = "Attack: True"

conv_ax.legend()
conv_ax.set_xlabel("Iteration")
conv_ax.set_ylabel("Convergence to Optimal")
conv_ax.set_ylim(bottom=0)
conv_ax.set_xlim(left=0)
conv_ax.grid(True)
conv_ax.set_title(f"{conv_title1}; {conv_title2}")
# conv_ax.set_ylim(0, 5)



###     PLOT: Trust Parameter

trust_plt = plt.figure()
trust_ax = trust_plt.add_subplot()
for ind in range(num_agents):
    for nbr_ind in range(num_agents):
        if (ind == nbr_ind) or (ind == ind_byz):
            continue
    
        color = "aquamarine"
        if nbr_ind == ind_byz:
            color = "orangered"
        
        trust_ax.plot(np.arange(iterations), trust_history[ind][nbr_ind, :].flatten(), label=f"({ind}, {nbr_ind})")#, color=color)

trust_ax.legend()
trust_ax.set_xlabel("Iterations")
trust_ax.set_ylabel("Trust Level")
trust_ax.set_ylim(bottom=0)
trust_ax.set_xlim(left=0)
trust_ax.grid(True)
trust_ax.set_title("Trust Parameter History")



###     PLOT: Lambda Norm

lam_plt = plt.figure()
lam_ax = lam_plt.add_subplot()
for ind in range(num_agents):
    for nbr_ind in range(num_agents):
        if (ind == nbr_ind) or (ind == ind_byz):
            continue
        
        color = "aquamarine"
        if nbr_ind == ind_byz:
            color = "orangered"
    
        lam_ax.plot(np.arange(iterations), lam_norm_history[ind][nbr_ind, :].flatten(), label=f"({ind}, {nbr_ind})", color=color)

lam_ax.legend()
lam_ax.set_xlabel("Iterations")
lam_ax.set_ylabel("Lambda Norm")
lam_ax.set_ylim(bottom=0)
lam_ax.set_xlim(left=0)
lam_ax.grid(True)
lam_ax.set_title("Lambda Norm History")

plt.show()