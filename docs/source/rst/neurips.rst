.. |icon| image:: /resource/icon/icon_pixel.png

.. figure:: /resource/image/splash.png

|icon| NeurIPS 2021 Dataset Track Supplementary Material
########################################################

`[Neural MMO v1.5 Trailer] <https://youtu.be/d1mj8yzjr-w>`_

This section is intended for reviewers evaluating Neural MMO for publication in NeurIPS 2021.

|icon| Versioning
-----------------

At time of writing, this website hosts the current v1.5.0 public release of Neural MMO. The manuscript submitted to NeurIPS covers the upcoming v1.5.1 release. The changes are fairly small, and the official release is slated for either late June or early July.

The most important differences from v1.5.0 are:
   - Increased the map size of our smaller setting from 60x60 to 128x128 to support an upcoming competition
   - Improved terrain generation to create a more diverse set of maps 
   - Modularized configuration options to support more types of research 
   - Changed spawning to place agents around the edges of large maps rather than at the center

The v1.5-competition branch contains code and trained models for the experiments in the paper. You can also build the current v1.5 release if you just want to try out the environment and associated tools.

|icon| Source Material
----------------------

Massively Multiplayer Online role-playing games simulate persistent virtual worlds with hundreds to thousands of concurrent players per server. They simpler than the real world but arguably much more complex than any game environment studied to date. MMOs support diverse gameplay, intentional specialization across different skills, moderately realistic economies with market forces driven by player interaction, and emergent ad-hoc social structures organized around various in-game activities. Unlike previously studied level- and round-based games, players in MMOs gain intrinsic abilities and tradable items which grant them various advantages that persist throughout hundreds to thousands of hours of play. This sort of long-term reasoning within a dynamic society of other intelligent agents is the default setting for real-world learning but notably absent in previously studied games. Neural MMO lacks the complexity of commercial MMOs developed by large professional studios, but it does adapt several key properties of the genre for computationally efficient research: long time horizons, large populations, and open-ended task specification, among others.

|icon| Achievement System
-------------------------

Neural MMO provides a reward of -1 for dying and 0 otherwise by default. As mentioned in the main text, users may define their own reward functions with full access to game state. We have recently begun experimenting with an achievement system (Fig. \ref{achievements}) that rewards agents for reaching gameplay milestones. For example, agents may receive a small reward for obtaining their first piece of armor, a medium reward for defeating three other players, and a large reward for traversing the entire map. The tasks and point values themselves are clearly domain-specific, but we believe this achievement system has several advantages compared to traditional reward shaping. First, since each task may be achieved only once, agents cannot simply farm one task repeatedly as sometimes occurs with poorly tuned dense rewards. Second, this property should make the achievement system less sensitive to the exact point tuning. Finally, attaining a high achievement score somewhat guarantees complex behavior since that is exactly what the point system was designed to represent. In practice, we have not yet observed a significant difference compared to training on lifetime alone. We have chosen to include this system in the Supplement nonetheless because it has been successful across a wide genre of human games, and we believe that this is likely to translate into a benefit for training and evaluation in the future.

.. figure:: /resource/image/achievements.png
Score points based on the highest tier task completed in each achievement category. We currently award 4, 10, and 25 points for easy, medium, and hard (top, middle, bottom) tasks respectively, for a total of 100 possible points.

|icon| Terrain Generation
-------------------------

Neural MMO includes a novel procedural content generation algorithm that creates maps with different terrain frequencies at different spacial locations. This means that environmental features are small and closely packed in some parts of the map and large but spread out in others. The core idea is simple: stretch the terrain output by one coherent noise algorithm (we use the Simplex noise implementation provided by the Python vec-noise package) using the output of another coherent noise map. The naive way of doing this is to use coherent noise to stretch the coordinates input to the second coherent noise generator. However, this produces very noticeable warping and spaghettification.

Our solution is substantially more complex but almost completely eliminates artifacting. First, we generate one noise map to define local frequency. We use standard multi-octave Simplex noise, but any similar algorithm would work. Second, we sample a large number of noise maps from a fixed frequency range. The precise number used is dependent on map size -- one octave per 8 tiles of resolution seems to work well. This means 16 octaves for 128x128 maps and 128 octaves for 1024x1024 maps, which is far more than is normally used in terrain generation. The idea is to use the local frequency map to select which of these octaves to blend at each point in space. We use the local frequency at each point to define the mean of a Gaussian distribution over octaves. Sampling one point per octave from this Gaussian produces the blend weightings.

Finally, we apply a trick to keep the terrain generation at the edges consistent. We bias the Gaussian mean using the distance from the center of the map. This creates a smooth blend between normal Simplex noise generation at the edges of the map (where consistent resource distributions are important to newly-spawned agents) and more visually interesting terrain at the center of the maps.

Other than producing more visually interesting maps than plain Simplex noise alone, performing this terrain generation has two benefits. First, it allows us to keep terrain generation at the edges of the map consistent. Second, it widens the training data distribution. The algorithm outlined above contains a smoothness parameter which we can sweep during terrain generation in order to produce more diverse maps that are not contained in the distribution of Simplex noise alone.

|icon| Spawning
---------------

All of our experiments hinge upon competitive incentives created by the presence of many concurrent agents. One of our experiments considers maximum population size as one such creator of competitive pressure, but there are other relevant factors as well. The exact mechanism by which agents are added to the environment is important. We considered two spawning algorithms throughout our experiments. Neither is strictly "better" than the other -- in general, continuous spawning is better suited to persistent simulation whereas concurrent spawning is better suited to round-based play.

**Continuous Spawning:** This spawning mechanism is inspired by traditional MMOs. The environment spawns new agents every game tick up to the population cap. Our experiments make three spawn attempts per game tick. For each spawn attempt, we select a tile around the edge of the map. If it is unoccupied, we spawn an agent. We also ensure that at least one agent is always present to avoid null observations.

Continuous spawning is useful because it dynamically sets the current population based on agent skill level. If agents die quickly, then fewer agents will be present at any given time, leaving more resources available. As agents learn to forage more efficiently, the increase in average survival time results in a larger population size. This produces greater population density at the edges of the map, which incentivizes agents to explore towards the center. However, it introduces the possibility of spawn camping: high-level agents can wait for more players to spawn and kill them immediately. This is bad because there is no possibility for newly-spawned agents, no matter how intelligent, to escape from this situation. Spawning agents around the edges of the entire map, as opposed to in one or a few dedicated areas (as is typically done in MMOs) helps disincentivize this behavior but does not fix the problem entirely.

**Concurrent Spawning:** This spawning mechanism is inspired by recent Battle Royale games. The environment spawns all agents at evenly spaced intervals along the edges of the map on the first game tick. No additional spawn attempts are made thereafter. If all agents die before hitting the simulation horizon, we sample a new map.

Concurrent spawning is useful because it is more fair to all players than continuous spawning. Some spawning locations are still better than others by virtue of proximity to more resources, but this method does at least eliminate advantages from asymmetric playtime and spawn camping.

|icon| Serialization
--------------------

MMOs are computationally efficient compared to most other game genres, meaning that the hardware required to simulate environments is much less than that required to simulate models. This allows us to develop in pure python with all the advantages therein. The only major downside is that computing observations by traversing the relevant python objects is still expensive. In fact, doing this naively is 50-100x slower than the entire rest of the environment. Serialization allows us to perform this computation around 50x faster. Our serialization scheme relies on the observation of each object being represented as a vector. This allows us to maintain a table where each row is an object and each attribute is an attribute. In practice, we maintain separate tables for each object type as well as for discrete and continuous attributes, but this is a minor implementation detail. The key point is that this representation enables us to compute observations by selecting rows from the table. We wrap all observable object classes with syntactic sugar that updates the table each time an attribute changes. We also maintain a grid of object IDs corresponding to tile and agent positions. Since agents observe a square crop of tiles around them, this grid will always contain the row indices corresponding to nearby objects. If we had chosen to represent maps in continuous space, a similar optimization would be possible using kD trees, but it would be slower by a logarithmic factor.
