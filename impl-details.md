# Implementation Details: O-RAN Dynamic Spectrum Allocation via Graph Coloring

Here is the comprehensive markdown compilation of the implementation details from the article to guide the coding and theoretical phases of your Algorithm Analysis (CAA) project.

---

## 1. Motivation and Problem Statement

### Why is Spectrum Allocation a Graph Problem?

In a 5G O-RAN network, a finite set of **Physical Resource Blocks (PRBs)** must be shared among many mobile users (UEs). Two users that would cause harmful interference to each other if assigned the same PRB are said to be in **conflict**. The core insight of this project is that this situation maps directly to the classic **graph coloring** problem:

| Graph Concept | Network Equivalent |
|---|---|
| Vertex $v \in V$ | An active User Equipment (UE) requesting service |
| Edge $(u, v) \in E$ | An interference conflict — these two UEs cannot share a PRB |
| Color $c \in \{1, \dots, P\}$ | A Physical Resource Block (PRB) index |
| Valid coloring | An interference-free PRB assignment where no two conflicting UEs share a PRB |

### Why Not Use an Exact Coloring Algorithm?

Graph coloring is **NP-hard** in the general case. Finding the minimum number of colors (the chromatic number) requires exponential time in the worst case. However, the xApp operates on a **sub-second timescale** (typically 1 ms to 500 ms scheduling intervals) and must produce a valid assignment for every time step. This makes exact algorithms (backtracking, branch-and-bound, or DSATUR) impractical for real-time operation.

Instead, the paper adopts a **greedy heuristic** inspired by the Welsh-Powell algorithm: sort vertices by decreasing weighted degree, then greedily assign the first available color. This produces a valid coloring in polynomial time (at most $O(U^2)$ in the worst case) and, in practice, uses close to the minimum number of colors for sparse to moderately dense interference graphs.

### Why is Fairness Needed on Top of Coloring?

Pure graph coloring treats all vertices equally — it simply tries to assign a color to as many vertices as possible. In a wireless network, this can lead to **user starvation**: UEs in dense areas (many conflicts) may never receive a PRB, while well-positioned UEs are always served. The paper addresses this with a **Modified Proportional Fair (MPF)** scheduling pass that runs after coloring:

- UEs left uncolored by the greedy pass get a second chance via a proportional-fair metric $M_u = (R_u / \bar{R}_u) \cdot w_u$.
- If an uncolored UE has a higher metric than a currently-served UE, it can **preempt** (steal) that PRB.
- This prevents indefinite starvation because UEs with low historical throughput receive boosted metrics over time.

### Summary of the Design Decomposition

The overall solution is a **two-phase pipeline**:

1. **Phase A — Graph Coloring (Algorithm 1):** Assign PRBs to as many UEs as possible without interference, using a greedy weighted-degree heuristic.
2. **Phase B — MPF Scheduling (Algorithm 2):** Resolve leftover uncolored UEs via proportional-fair preemption, ensuring no user is permanently starved.

This decomposition separates the correctness concern (no interference) from the fairness concern (no starvation), making the system both correct and equitable.

---

## 2. Design Technique: Greedy Graph Coloring with Fairness Post-Processing

### Choice of Algorithmic Technique

This project applies **Graph Algorithms** — specifically, a constructive greedy heuristic for the vertex coloring problem. The technique falls under the broader category of **graph-theoretic optimization**, which is a classic algorithm design paradigm alongside Divide and Conquer, Dynamic Programming, and Greedy Strategies.

The algorithm follows the **greedy approach**:

1. Make a locally optimal choice at each step (assign the lowest-indexed available PRB to the current UE).
2. Never reconsider past decisions (no backtracking).
3. Use a **smart ordering heuristic** (Welsh-Powell: sort by descending weighted degree) to improve the quality of the greedy solution.

### Why Welsh-Powell Ordering?

The Welsh-Powell theorem guarantees that the greedy coloring algorithm uses at most $\Delta + 1$ colors, where $\Delta$ is the maximum vertex degree. By processing high-degree vertices first, we ensure that the most constrained UEs (those with many conflicts) get priority access to the full set of available PRBs, while less-constrained UEs can more easily adapt to whatever PRBs remain.

### How the Two Algorithms Work Together

```
┌─────────────────────────────────────────────────┐
│            SCHEDULING TIME STEP t                │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. BUILD CONFLICT GRAPH G(t)                    │
│     For each pair (u, v):                        │
│       - Same O-RU? → add edge                    │
│       - SINR too low if sharing? → add edge      │
│     Complexity: O(U²) pairwise scan              │
│                                                  │
│  2. SORT UEs BY WEIGHTED DEGREE                  │
│     deg(u) × w_u, descending                     │
│     Complexity: O(U log U)                       │
│                                                  │
│  3. ALGORITHM 1: GREEDY GRAPH COLORING           │
│     For each UE in sorted order:                 │
│       Assign first available PRB (color)         │
│       not used by any neighbor                   │
│     Complexity: O(U · Δ) where Δ = max degree    │
│     → O(U log U + E) average, O(U²) worst       │
│                                                  │
│  4. ALGORITHM 2: MPF SCHEDULING                  │
│     For each uncolored UE:                       │
│       Compute MPF metric per candidate PRB       │
│       Preempt if M_u > M_owner                   │
│     Complexity: O(U_uncolored · P · Δ)           │
│                                                  │
│  5. UPDATE METRICS & LOG                         │
│     EWMA throughput, Jain's fairness, CSV log    │
│                                                  │
└─────────────────────────────────────────────────┘
```

### Connecting the Simulation Phases to Complexity Analysis

The experiment is designed to validate the theoretical complexity bounds through three controlled scenarios:

| Phase | Scenario | Graph Density | Expected Complexity | How It Is Achieved |
|---|---|---|---|---|
| **Phase 1** (Average) | Random UE positions with mobility | Sparse–moderate | $O(U \log U + E)$ | UEs spread across 3 O-RU coverage areas; conflict edges depend on proximity |
| **Phase 2** (Worst) | All UEs colocated at origin | Dense/complete | $O(U^2)$ | All UEs connect to the same Macro O-RU → every pair conflicts → $E = U(U-1)/2$ |
| **Phase 3** (Best) | UEs on 700 m grid | Nearly empty | $\Omega(U \log U)$ | 700 m spacing ensures UEs are too far apart to interfere → $E \approx 0$; only the sort matters |

By plotting execution time versus UE count for each phase, the experiment empirically demonstrates the quadratic growth of the worst case, the near-linear growth of the best/average cases, and the role of edge density $E$ in bridging the gap between them.

---

## 3. Problem Modeling: The Interference Graph $G(t)$
To implement the solution, you must first construct a dynamic user-centric conflict graph $G(t) = (V(t), E(t))$ representing the network at a given time $t$. 
*   **Vertices $V(t)$:** Each vertex represents an active User Equipment (UE) in the network.
*   **Colors $P$:** The available colors represent the Physical Resource Blocks (PRBs) to be allocated.
*   **Edges $E(t)$:** An undirected edge $(u, v)$ represents an interference conflict where two users cannot share the same PRB. An edge is created in your code if:
    1.  The users are connected to the same Radio Unit (RU).
    2.  The users are on different RUs, but assigning the same PRB to both causes severe inter-cell interference, dropping their achievable data rate below their specific satisfaction tolerance margin.

## 4. Core Algorithm: Policy-Guided Graph Coloring
The main algorithm executed at the Near-RT RIC (xApp) is a heuristic weighted graph coloring approach (based on the Welsh-Powell or Greedy coloring logic). 

**Algorithm 1: Policy-Guided Graph Coloring**
*   **Input:** Conflict graph $G(t)$, user priority weights $\{w_u\}$, and available PRBs $P$.
*   **Step 1:** Sort all UEs in descending order based on their weighted degree (number of conflicts).
*   **Step 2:** Loop through each UE $u$ in the sorted list.
*   **Step 3:** Loop through each available PRB $p \in P$.
*   **Step 4:** Check if PRB $p$ conflicts with any of the already colored neighbors of UE $u$.
*   **Step 5:** If there is no conflict, assign $c_u \leftarrow p$, break the inner loop, and move to the next UE.
*   **Output:** A colored graph $G'(t)$ representing the interference-free PRB allocation.

## 5. Post-Coloring Fairness: Modified Proportional Fair (MPF)
Because graph coloring might leave some UEs without PRBs (uncolored) due to severe conflicts, the authors implement an MPF scheduling algorithm to guarantee time-sharing and fairness.

**Algorithm 2: MPF Scheduling**
*   **Input:** Colored graph $G'(t)$, user weights $\{w_u\}$, PRBs $P$.
*   **Step 1:** For each uncolored UE $u$, compute the MPF metric over candidate PRBs: $M_u(t) = \frac{R_{u,p}(t)}{\bar{R}_u(t)} \cdot w_u$. 
    *   *Note:* $R_{u,p}(t)$ is the instantaneous achievable rate, and $\bar{R}_u(t)$ is the exponentially weighted moving average (EWMA) throughput.
*   **Step 2:** Find the PRB $p^*$ that yields the highest MPF metric for the uncolored user.
*   **Step 3:** If $p^*$ is currently assigned to a colored UE $v$, compare their metrics. If $M_u(t) > M_v(t)$, steal the PRB and schedule UE $u$ on $p^*$ instead of $v$.
*   **Step 4:** Update the EWMA throughput $\bar{R}_u(t)$ for all users using the smoothing factor $\alpha$ (e.g., $\alpha = 0.1$).

## 6. Theoretical Complexity Analysis
Based on our previous mathematical deductions for your report, the implementation of Algorithm 1 yields the following time complexities:
*   **Best Case:** $\Omega(U \log U)$. Occurs when users are geographically dispersed with no interference (0 edges). The algorithm only spends time sorting the vertices.
*   **Worst Case:** $O(U^2)$. Occurs in extremely dense networks where every user interferes with everyone else (a complete graph). The inner loops must check every connection.
*   **Average Case:** $O(U \log U + E)$. The standard network scenario where sorting the users and traversing the existing conflict edges dictate the performance.

## 7. Experimental Setup and Simulation Parameters
To build your experiment file in Python or C++ and test your algorithm, you should recreate the environment parameters used by the authors:
*   **Network Topology:** 3 O-RUs (1 Macro O-RU and 2 Micro O-RUs). 
*   **Coordinates:** The Macro cell is at position `(0,0)`, and the two Micro cells are at `(200,0)` and `(-200,0)`.
*   **Coverage Radii:** Macro cell radius is 300m; Micro cell radius is 50m.
*   **Path-loss Exponents:** Macro $\alpha_M = 2.7$ ; Micro $\alpha_m = 2.8$.
*   **Mobility Model:** UEs move using a 2D random-walk model with a constant speed of 1.5 m/s.
*   **UE Demands:** Users randomly demand 0.5 Mbps, 1 Mbps, or 1.5 Mbps.
*   **Priorities and Tolerance:** For baseline testing, set all user priority weights $w_u = 1$ and satisfaction tolerance margin $\eta_u = 0$ (meaning users demand 100% of their requested rate).