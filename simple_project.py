# Full library imports
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

# Function imports
from tqdm import tqdm
from copy import deepcopy
from math import factorial



### Initializations

x_dim = 3
y_dim = 3
iterations = 100
init_guess = 0
np.random.seed(6052)
add_attack = True
attack_center = 0
attack_scale = 1
adaptive_penalty = True
trust_parameter_scale = 1e1
rho = 1
trust_threshold = 0.2
normalization = False
show_byz_plt = False

show_prob1 = False
show_prob2 = False

# Honest and Rogue Agents
num_agents = 10
num_byz = 3
# ind_byz = np.random.randint(0, num_agents, num_byz)
ind_byz = np.random.choice(np.arange(0, num_agents), size=num_byz, replace=False)
print("Byzantine Agents:", ind_byz)

# Color Array
colors = ['darkgray', 'deeppink', 'springgreen',
          'aquamarine', 'crimson', 'slateblue',
          'mediumpurple', 'darkviolet', 'violet',
          'fuchsia', 'olivedrab', 'steelblue']

# Edge List
edges = [   (0, 1), (0, 2), (0, 3),
            (0, 4), (0, 6), (1, 2),
            (1, 3), (1, 6), (2, 4),
            (2, 6), (2, 7), (2, 9),
            (3, 4), (3, 5), (3, 6),
            (4, 5), (4, 7), (5, 6),
            (5, 7), (5, 9), (6, 7),
            (6, 8), (7, 8), (7, 9),
            (8, 9), (0, 8), (1, 5),
            (4, 8)]
num_edges = len(edges)

arcs = deepcopy(edges)
num_arcs = 2*num_edges

for i in range(num_edges):
    this_edge = edges[i]
    new_edge = (this_edge[1], this_edge[0])
    edges.append(new_edge)

# Neighbor List - Agents are connected to each other
nbr_list = [None] * num_agents
for i in range(num_agents):
    nbr_list_i = []
    for j in range(num_arcs):
        this_arc = edges[j]
        if this_arc[0] == i:
            nbr_list_i.append(this_arc[1])
    nbr_list[i] = nbr_list_i
    # print("Agent", i, "neighbors:", nbr_list_i)

# Trust Function
def trust_parameter(param):
    trust = rho * trust_parameter_scale / ((np.linalg.norm(param)) + trust_parameter_scale)
    # trust = 1 - np.linalg.norm(param) / trust_parameter_scale
    # trust = trust_parameter_scale * (np.exp(+np.linalg.norm(param)) + 1, -1)
    return trust

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



### Centralized: Finding optimal for each local objective function

# print("\nOptimal x for obj function i:")
local_opt_x = []
local_opt_x_hon = []
for i in range(num_agents):
    x_cp = cp.Variable((x_dim, 1))
    obj_func = 0.5 * cp.power(cp.norm(y_arr[i] - M_arr[i] @ x_cp), 2)
    prob_i = cp.Problem(cp.Minimize(obj_func))
    
    prob_i.solve()
    x_opt_i = np.array(x_cp.value).reshape((-1, 1))
    local_opt_x.append(x_opt_i)
    if i not in ind_byz:
        local_opt_x_hon.append(x_opt_i)
    # print("\t", x_opt_i.flatten())



### Centralized: Finding optimal for honest agents

x_cp = cp.Variable((x_dim, 1))
obj_func = 0
for i in range(num_agents):
    if i in ind_byz:
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
    x_arr.append(init_guess * np.ones((x_dim, 1)))

# Define local variables for edges
lam_arr = [None] * num_agents
tot_div_arr = [None] * num_agents
# z_cp_arr = [None] * num_agents
# z_arr = [None] * num_agents

for i in range(num_agents):
    lam_arr_i = {}
    tot_div_arr_i = {}
    # z_cp_arr_i = {}
    # z_arr_i = {}
    
    for j in nbr_list[i]:
        lam_arr_i[j] = np.zeros((x_dim, 1))
        tot_div_arr_i[j] = np.zeros((x_dim, 1))
        # z_cp_arr_i[j] = cp.Variable((x_dim, 1))
        # z_arr_i[j] = np.zeros((x_dim, 1))
    
    lam_arr[i] = lam_arr_i
    tot_div_arr[i] = tot_div_arr_i
    # z_cp_arr[i] = z_cp_arr_i
    # z_arr[i] = z_arr_i

# Define Trust Variable
trust = np.ones((num_agents, num_agents))
for i in range(num_agents):
    for j in range(num_agents):
        if i == j:
            trust[i][j] = 0
            
if normalization:
    trust = trust / num_agents

# Storage Variables
x_norm_diff_history = [np.zeros(iterations) for i in range(num_agents)]
x_history = [np.zeros((x_dim, iterations)) for i in range(num_agents)]
trust_history = [np.zeros((num_agents, iterations)) for i in range(num_agents)]
lam_norm_history = [np.zeros((num_agents, iterations)) for i in range(num_agents)]
func_val_history = np.zeros(iterations)

# ADMM Iterations

for iter in tqdm(range(iterations), desc="Iterations", leave=False):
    # Store Variables
    this_x_arr = deepcopy(x_arr)
    func_val = 0
    for ind in range(num_agents):
        for nbr_ind in nbr_list[ind]:
            assert (ind != nbr_ind)
            lam_norm_history[ind][nbr_ind, iter] = np.linalg.norm(lam_arr[ind][nbr_ind])
        if (ind not in ind_byz):
            func_val += 0.5 * np.power(np.linalg.norm(y_arr[ind] - M_arr[ind] @ this_x_arr[ind]), 2)
    func_val_history[iter] = func_val
    
    # x-Update
    for ind in range(num_agents):
        obj_i = 0.5 * cp.power(cp.norm(y_arr[ind] - M_arr[ind] @ x_cp_arr[ind]), 2)
        
        for nbr_ind in nbr_list[ind]:
            aij = 1
            if adaptive_penalty and (ind not in ind_byz):
                aij = trust[ind][nbr_ind] #trust_parameter(tot_div_arr[ind][nbr_ind])
            nbr_x_arr = this_x_arr[nbr_ind]
            if add_attack and (nbr_ind in ind_byz):
                # nbr_x_arr += np.random.uniform(attack_center - attack_scale, attack_center + attack_scale)
                nbr_x_arr += attack_center + iter*attack_scale
            # obj_i += x_cp_arr[ind].T @ (lam_arr[ind][nbr_ind] - aij*(this_x_arr[ind] + this_x_arr[nbr_ind]))
            obj_i += x_cp_arr[ind].T @ (lam_arr[ind][nbr_ind] - aij*(this_x_arr[ind] + nbr_x_arr))
            obj_i += aij * cp.power(cp.norm(x_cp_arr[ind]), 2)
        
        # Solve
        prob1 = cp.Problem(cp.Minimize(obj_i), [])
        try:
            prob1.solve(solver=cp.SCS, verbose=show_prob1)
        except cp.SolverError:
            print("Solver Error @ Prob 1")
            quit()
        
        # Byzantine Agent Conditional
        new_x = deepcopy(np.array(x_cp_arr[ind].value).reshape((-1, 1)))
        # if add_attack and (ind in ind_byz):
        #     # new_x += attack_center
        #     new_x += np.random.uniform(attack_center - attack_scale, attack_center + attack_scale)
        #     # new_x += np.random.normal(attack_center, attack_scale)
            
        # Store
        x_arr[ind] = new_x
        x_history[ind][:, iter] = new_x.flatten()
        x_norm_diff = np.linalg.norm(new_x - x_opt_all)
        if add_attack:
            x_norm_diff = np.linalg.norm(new_x - x_opt_honest)
        x_norm_diff_history[ind][iter] = x_norm_diff


    # lam-Update
    for ind in range(num_agents):
        
        tot_trust_i = 0
        
        for nbr_ind in nbr_list[ind]:
            aij = 1
            if adaptive_penalty and (ind not in ind_byz):
                aij = trust[ind][nbr_ind] #trust_parameter(tot_div_arr[ind][nbr_ind]) # lam_arr[ind][nbr_ind]
            trust_history[ind][nbr_ind, iter] = aij
                
            constr_dev = (x_arr[ind] - x_arr[nbr_ind])
            lam_arr[ind][nbr_ind] += (aij*constr_dev)
            tot_div_arr[ind][nbr_ind] += aij*np.power(np.linalg.norm(rho*constr_dev), 2)
            
            new_trust = trust_parameter(tot_div_arr[ind][nbr_ind])
            # new_trust = trust_parameter(lam_arr[ind][nbr_ind])
            if new_trust < trust_threshold:
                new_trust = 0
            trust[ind][nbr_ind] = new_trust
            tot_trust_i += new_trust
        
        # Normalize Trust
        if normalization:
            trust[ind][:] = trust[ind][:] / tot_trust_i


# Print ADMM Results
# print("\nADMM: local x at each agent")
# for i in range(num_agents):     
    # print(f"\tAgent {i}: {x_arr[i].flatten()}")

# print("\nADMM: local norm(lam) for each agent")
for i in range(num_agents):
    norm_lam_i = []
    for j in nbr_list[i]:
        if i == j:
            norm_lam_i.append(0)
        else: 
            norm_lam_i.append(float(np.linalg.norm(lam_arr[i][j])))
    # print(f"\tAgent {i}: {norm_lam_i}")
    
# print("\nADMM: local trust param for each agent")
for i in range(num_agents):
    norm_tot_div_i = []
    for j in nbr_list[i]:
        if i == j:
            norm_tot_div_i.append(0)
        else: 
            norm_tot_div_i.append(float(trust_parameter(tot_div_arr[i][j])))
    # print(f"\tAgent {i}: {norm_tot_div_i}")


def in_hull (p_arr, hull_arr):
    from scipy.spatial import Delaunay
    p = np.hstack(p_arr).T.reshape((-1, x_dim))
    hull = np.hstack(hull_arr).T.reshape((-1, x_dim))
    # print(p)
    # print(hull)
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    
    return hull.find_simplex(p) >= 0


print("\nADMM: local solution within convex hull")
hull_bool_arr = in_hull(x_arr, local_opt_x_hon)
for i in range(num_agents):
    print(f"\tAgent {i+1}: {hull_bool_arr[i]}")


# print("\nCentralized: global honest solution within convex hull")
# print(f"\n\tGlobal Honest Sol.: {in_hull(x_opt_honest, local_opt_x_hon)}")

###     PLOT: Convergence Results
plot_conv_bool = True
if plot_conv_bool:
    conv_plt = plt.figure()
    conv_ax = conv_plt.add_subplot()
    for i in range(num_agents):
        
        if (i in ind_byz) and show_byz_plt:
            color = 'orangered'
            conv_ax.plot(np.arange(iterations), x_norm_diff_history[i], label=f"Agent {i+1}", color=color)
            
        elif (i not in ind_byz): 
            color = colors[i]
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


###     PLOT: Individual Convergence Results

plot_ind_cov_bool = False
if plot_ind_cov_bool:
    for var_i in range(x_dim):
        conv_plt = plt.figure()
        conv_ax = conv_plt.add_subplot()
        
        for i in range(num_agents):
            x_goal = x_opt_all[var_i]
            if add_attack:
                x_opt_honest[var_i]
                
            # Byzantine Agent
            if (i in ind_byz) and show_byz_plt: 
                color = 'orangered'
                x_hist = np.abs(x_history[i][var_i, :].flatten() - x_goal)
                conv_ax.plot(np.arange(iterations), x_hist, label=f"Agent {i+1}", color=color, linestyle='solid')

            # Normal Agent
            elif (i not in ind_byz): 
                color = colors[i]
                x_hist = np.abs(x_history[i][var_i, :].flatten() - x_goal)
                conv_ax.plot(np.arange(iterations), x_hist, label=f"Agent {i+1}", color=color, linestyle='solid')

        conv_title1 = "Adaptive Penalty: False"
        if adaptive_penalty:
            conv_title1 = "Adaptive Penalty: True"
            
        conv_title2 = "Attack: False"
        if add_attack:
            conv_title2 = "Attack: True"

        conv_ax.legend()
        conv_ax.set_xlabel("Iteration")
        conv_ax.set_ylabel(f"| x_{var_i+1}(Reg. ADMM) - x_{var_i+1}(Mod. ADMM) |")
        conv_ax.set_xlim(left=0)
        conv_ax.set_ylim(bottom=0)
        conv_ax.grid(True)
        conv_ax.set_title(f"x_{var_i+1}; {conv_title1}; {conv_title2}")



###     PLOT: Trust All Parameters (over edges)

plot_all_trust_bool = True
if plot_all_trust_bool:
    trust_plt = plt.figure()
    trust_ax = trust_plt.add_subplot()
    for ind in range(num_agents):
        for nbr_ind in range(num_agents):
            if (ind == nbr_ind) or (ind in ind_byz):
                continue
        
            color = "aquamarine"
            if nbr_ind in ind_byz:
                color = "orangered"
            
            trust_ax.plot(np.arange(iterations), trust_history[ind][nbr_ind, :].flatten(), label=f"({ind}, {nbr_ind})", color=color)

    trust_ax.legend()
    trust_ax.set_xlabel("Iterations")
    trust_ax.set_ylabel("Trust Level")
    trust_ax.set_ylim(bottom=0)
    trust_ax.set_xlim(left=0)
    trust_ax.grid(True)
    trust_ax.set_title("Trust Parameter History")


###     PLOT: Aggregated Trust (averaged for agents)

plot_avg_trust_bool = False
# plt.style.use('dark_background')
if plot_avg_trust_bool:
    avg_trust_plt = plt.figure()
    avg_trust_ax = avg_trust_plt.add_subplot()
    # avg_trust_ax.set_facecolor('dimgray')
    for ind in range(num_agents):
        avg_trust = np.zeros(iterations)
        
        for nbr_ind in range(num_agents):
            if (ind == nbr_ind):
                continue
                
            avg_trust += trust_history[nbr_ind][ind, :].flatten() / (num_agents - 1)
        
        color = "aquamarine"
        if ind in ind_byz:
            color = "orangered"
            
        avg_trust_ax.plot(np.arange(iterations), (1 - avg_trust), label=f"Agent {ind}", color=color)
    
    avg_trust_ax.plot(np.arange(iterations), 0.5*np.ones(iterations), linestyle="--", color="black")
    
    avg_trust_ax.legend(loc='center right')
    avg_trust_ax.set_xlabel("Iterations")
    avg_trust_ax.set_ylabel("Belief of Byzantine Agent")
    avg_trust_ax.set_ylim(bottom=0, top=1)
    avg_trust_ax.set_xlim(left=0, right=(iterations-1))
    avg_trust_ax.grid(True)
    avg_trust_ax.set_title("Byzantine Detector")


###     PLOT: Lambda Norm

plot_lam_norm_bool = False
if plot_lam_norm_bool:
    lam_plt = plt.figure()
    lam_ax = lam_plt.add_subplot()
    for ind in range(num_agents):
        for nbr_ind in range(num_agents):
            if (ind == nbr_ind) or (ind in ind_byz):
                continue
            
            color = "aquamarine"
            if nbr_ind in ind_byz:
                color = "orangered"
        
            lam_ax.plot(np.arange(iterations), lam_norm_history[ind][nbr_ind, :].flatten(), label=f"({ind}, {nbr_ind})", color=color)

    lam_ax.legend()
    lam_ax.set_xlabel("Iterations")
    lam_ax.set_ylabel("Lambda Norm")
    lam_ax.set_ylim(bottom=0)
    lam_ax.set_xlim(left=0)
    lam_ax.grid(True)
    lam_ax.set_title("Lambda Norm History")



###     PLOT: Consensus Constraints

# Array generation

constr_history = []
constr_edge_tracker = []

for i in range(num_agents):
    for j in nbr_list[i]:
        if (j <= i): # Continue if edge constraint is already included or if self-constraint
            continue
        
        this_constr_hist = np.zeros(iterations)
        for iter in range(iterations):
            constr = (x_history[i][:, iter].flatten() - x_history[j][:, iter].flatten())
            constr_norm = np.linalg.norm(constr)
            this_constr_hist[iter] = constr_norm
            
        constr_history.append(this_constr_hist)
        constr_edge_tracker.append((i, j))

# Plotting

plot_constr_bool = False
if plot_constr_bool:
    constr_plt = plt.figure()
    constr_ax = constr_plt.add_subplot()
    for ind, this_constr_hist in enumerate(constr_history):
        # ID
        curr_id = constr_edge_tracker[ind][0]
        nbr_id = constr_edge_tracker[ind][1]
        
        # Skip
        if (curr_id in ind_byz):
            continue
        
        # Color
        color = "aquamarine"
        if (nbr_id in ind_byz):
            color = "orangered"
        
        constr_ax.plot(np.arange(iterations), this_constr_hist, label=f"{constr_edge_tracker[ind]}", color=color)

    constr_ax.legend()
    constr_ax.set_xlabel("Iterations")
    constr_ax.set_ylabel("Constraint Norm")
    constr_ax.set_ylim(bottom=0)
    constr_ax.set_xlim(left=0)
    constr_ax.grid(True)
    constr_ax.set_title("Constraint Norm History")


###     PLOT: Function Value

plot_fval_bool = True
if plot_fval_bool:
    fval_plt = plt.figure()
    fval_ax = fval_plt.add_subplot()
    fval_ax.semilogy(np.arange(iterations), func_val_history)
    
    fval_ax.set_xlabel("Iterations")
    fval_ax.set_ylabel("Function Value")
    fval_ax.set_ylim(bottom=0)
    fval_ax.set_xlim(left=0)
    fval_ax.grid(True)
    fval_ax.set_title("Function Value History")
    

plt.show()