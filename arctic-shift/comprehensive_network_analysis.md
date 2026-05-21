# ETHZ Reddit Network Analysis: Methods and Results

This document summarizes the structural and social network analyses performed on the `r/ethz` user and subreddit datasets. The analysis is broken down into two distinct networks: the **User Interaction Network** and the **Subreddit Co-Participation Network**.

---

## Part 1: User Interaction Network

### Methodology
To understand how users within the ETH Zurich Reddit community interact, we built a directed, weighted graph where:
- **Nodes** represent individual Reddit users who have been classified with a specific "voice type" (e.g., `decentral_individual` for students/staff, `external_individual` for outsiders).
- **Edges** represent interactions between users. An edge is formed through three mechanisms:
  1. **Replies:** User A replies directly to User B's post or comment.
  2. **Mentions:** User A mentions `u/UserB` in a comment.
  3. **Co-participation:** User A and User B both comment on the exact same post.
- **Edge Weights** are the total count of these interactions between a pair of users.
- **Algorithms Used:**
  - **Louvain Heuristic:** Used to detect communities (clusters of heavily interacting users).
  - **PageRank:** Used to calculate the network influence and centrality of each user.
  - **Attribute Assortativity:** Used to calculate homophily (the tendency for users to interact with others of the exact same voice type).

### Key Results
- **Graph Size:** 4,242 users and 421,009 interactions.
- **Modularity (0.2429):** The modularity score is quite low. Over 35.8% of interactions cross community boundaries. Rather than forming isolated groups, the network is one giant, densely connected "hairball".
- **Lack of Echo Chambers:** The Assortativity Coefficient based on voice type is **0.0179** (on a scale from -1 to 1). This score is extremely close to zero, meaning users have no preference for interacting with their own kind. Students talk to external individuals, applicants talk to former students, and everyone talks to unknowns.
- **Institutional Influence:** Institutional voices (`decentral`) are incredibly central. Their average PageRank is **0.000712**, which is nearly three times higher than individual students (`decentral_individual` at 0.000226). When official accounts speak, they act as major informational hubs.
- **Super-Spreaders:** The network is heavily dependent on a few "power users". The top user (`BNI_sp`) has an interaction weight of over 5,900. These users bridge the entire network together.

---

## Part 2: Subreddit Co-Participation Network

### Methodology
To understand the broader Reddit context of the ETHZ community, we built an undirected, weighted graph where:
- **Nodes** represent Subreddits where our classified users have been active.
- **Edges** are formed when the same classified user posts/comments in both Subreddit A and Subreddit B.
- **Edge Weights** represent the total number of unique classified users shared between the two subreddits (minimum 2 shared users to form an edge).
- **Algorithms Used:**
  - **Louvain Heuristic:** Used to detect thematic clusters of subreddits.
  - **PageRank & Weighted Degree Centrality:** Used to determine which subreddits act as the biggest "bridges" for our users.

### Key Results
- **Graph Size:** 2,573 subreddits and 56,229 shared user connections.
- **Modularity (0.1870):** The subreddit network is even more heavily mixed than the user network. Users from the ETHZ dataset jump across thousands of different subreddits.
- **Thematic Communities:** Despite the low modularity, the Louvain algorithm perfectly clustered the subreddits into 4 distinct thematic categories:
  1. **Academic & Career (988 subreddits):** Anchored around `r/ethz`, `r/EPFL`, `r/PhD`, `r/csMajors`, and `r/gradadmissions`. This community naturally has the highest density of current students (`decentral_individual`).
  2. **Swiss Life & Finance (779 subreddits):** Anchored by `r/Switzerland`, `r/askswitzerland`, `r/zurich`, and `r/SwissPersonalFinance`.
  3. **Viral & Entertainment (457 subreddits):** Filled with general Reddit time-wasters like `r/mildlyinfuriating`, `r/interestingasfuck`, and `r/meirl`.
  4. **European / DACH Region (349 subreddits):** Focused on geography and the broader German-speaking world (`r/europe`, `r/germany`, `r/de`, `r/Studium`, `r/ich_iel`).
- **`r/ethz` is the Ultimate Bridge:** `r/ethz` has the absolute highest PageRank (0.0619) and Degree weight in the entire graph. It serves as the primary conduit connecting the academic subreddits, the Swiss national subreddits, and the international/entertainment subreddits for this specific group of users. 

> [!TIP]
> **Data Visualization Takeaway**
> Because `r/Switzerland` and `r/askswitzerland` drive so many interactions between your users, the physics simulations for the User Interaction Graph will naturally clump everything together. To cleanly visualize just the academic network, filter the user interactions to strictly those occurring on `r/ethz`.
