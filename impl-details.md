# Implementation Details: O-RAN Dynamic Spectrum Allocation via Graph Coloring

Here is the comprehensive markdown compilation of the implementation details from the article to guide the coding and theoretical phases of your Algorithm Analysis (PCA) project.

## 1. Problem Modeling: The Interference Graph $G(t)$
To implement the solution, you must first construct a dynamic user-centric conflict graph $G(t) = (V(t), E(t))$ representing the network at a given time $t$. 
*   **Vertices $V(t)$:** Each vertex represents an active User Equipment (UE) in the network.
*   **Colors $P$:** The available colors represent the Physical Resource Blocks (PRBs) to be allocated.
*   **Edges $E(t)$:** An undirected edge $(u, v)$ represents an interference conflict where two users cannot share the same PRB. An edge is created in your code if:
    1.  The users are connected to the same Radio Unit (RU).
    2.  The users are on different RUs, but assigning the same PRB to both causes severe inter-cell interference, dropping their achievable data rate below their specific satisfaction tolerance margin.

## 2. Core Algorithm: Policy-Guided Graph Coloring
The main algorithm executed at the Near-RT RIC (xApp) is a heuristic weighted graph coloring approach (based on the Welsh-Powell or Greedy coloring logic). 

**Algorithm 1: Policy-Guided Graph Coloring**
*   **Input:** Conflict graph $G(t)$, user priority weights $\{w_u\}$, and available PRBs $P$.
*   **Step 1:** Sort all UEs in descending order based on their weighted degree (number of conflicts).
*   **Step 2:** Loop through each UE $u$ in the sorted list.
*   **Step 3:** Loop through each available PRB $p \in P$.
*   **Step 4:** Check if PRB $p$ conflicts with any of the already colored neighbors of UE $u$.
*   **Step 5:** If there is no conflict, assign $c_u \leftarrow p$, break the inner loop, and move to the next UE.
*   **Output:** A colored graph $G'(t)$ representing the interference-free PRB allocation.

## 3. Post-Coloring Fairness: Modified Proportional Fair (MPF)
Because graph coloring might leave some UEs without PRBs (uncolored) due to severe conflicts, the authors implement an MPF scheduling algorithm to guarantee time-sharing and fairness.

**Algorithm 2: MPF Scheduling**
*   **Input:** Colored graph $G'(t)$, user weights $\{w_u\}$, PRBs $P$.
*   **Step 1:** For each uncolored UE $u$, compute the MPF metric over candidate PRBs: $M_u(t) = \frac{R_{u,p}(t)}{\bar{R}_u(t)} \cdot w_u$. 
    *   *Note:* $R_{u,p}(t)$ is the instantaneous achievable rate, and $\bar{R}_u(t)$ is the exponentially weighted moving average (EWMA) throughput.
*   **Step 2:** Find the PRB $p^*$ that yields the highest MPF metric for the uncolored user.
*   **Step 3:** If $p^*$ is currently assigned to a colored UE $v$, compare their metrics. If $M_u(t) > M_v(t)$, steal the PRB and schedule UE $u$ on $p^*$ instead of $v$.
*   **Step 4:** Update the EWMA throughput $\bar{R}_u(t)$ for all users using the smoothing factor $\alpha$ (e.g., $\alpha = 0.1$).

## 4. Theoretical Complexity Analysis
Based on our previous mathematical deductions for your report, the implementation of Algorithm 1 yields the following time complexities:
*   **Best Case:** $\Omega(U \log U)$. Occurs when users are geographically dispersed with no interference (0 edges). The algorithm only spends time sorting the vertices.
*   **Worst Case:** $O(U^2)$. Occurs in extremely dense networks where every user interferes with everyone else (a complete graph). The inner loops must check every connection.
*   **Average Case:** $O(U \log U + E)$. The standard network scenario where sorting the users and traversing the existing conflict edges dictate the performance.

## 5. Experimental Setup and Simulation Parameters
To build your experiment file in Python or C++ and test your algorithm, you should recreate the environment parameters used by the authors:
*   **Network Topology:** 3 O-RUs (1 Macro O-RU and 2 Micro O-RUs). 
*   **Coordinates:** The Macro cell is at position `(0,0)`, and the two Micro cells are at `(200,0)` and `(-200,0)`.
*   **Coverage Radii:** Macro cell radius is 300m; Micro cell radius is 50m.
*   **Path-loss Exponents:** Macro $\alpha_M = 2.7$ ; Micro $\alpha_m = 2.8$.
*   **Mobility Model:** UEs move using a 2D random-walk model with a constant speed of 1.5 m/s.
*   **UE Demands:** Users randomly demand 0.5 Mbps, 1 Mbps, or 1.5 Mbps.
*   **Priorities and Tolerance:** For baseline testing, set all user priority weights $w_u = 1$ and satisfaction tolerance margin $\eta_u = 0$ (meaning users demand 100% of their requested rate).