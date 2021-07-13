.. |icon| image:: /resource/icon/icon_pixel.png

.. figure:: /resource/image/splash.png

|icon| NeurIPS 2021 Dataset Track Supplementary Material
########################################################

`[Neural MMO v1.5 Trailer] <https://youtu.be/d1mj8yzjr-w>`_

:download:`[Supplemental Tables and Discussion of Training, Architecture, etc.] </resource/update/neurips_supplement.pdf>` 

This section is intended for reviewers evaluating Neural MMO for publication in NeurIPS 2021.

|icon| Author Statement
-----------------------

The Neural MMO environment, renderer, associated tools, baseline models, and all associated training code can be found in the Github repositories linked from the project site, **neuralmmo.github.io**. This includes branches for all prior versions of the environment, and the entire commit history post v1.0 is public. We release these materials under the permissive MIT license and bear all responsibility in the case of violation of rights. The project page additionally includes documentation, API references, tutorials, path history, contributor acknowledgement, and links to all past papers, slides, recorded talks, and other materials. By the release of v1.5.1 (see below), we will update the baselines on this site to include dashboard images, performance tables, pretrained models, and one-line evaluation/training commands for reproducing the experiments in this paper. These are already available for v1.5.

**Ethics:** Neural MMO is designed specifically for fundamental research rather than for immediate practical use. As a basic research platform, Neural MMO could be adopted for positive and negative applications. We will monitor surrounding work on the platform and address any potential misuse that may arise.

|icon| Versioning
-----------------

At time of writing, this website hosts the current v1.5.0 public release of Neural MMO. The manuscript submitted to NeurIPS covers the upcoming v1.5.1 release. The changes are fairly small, and the official release is slated for either late June or early July.

The most important differences from v1.5.0 are:
   - Increased the map size of our smaller setting from 60x60 to 128x128 to support an upcoming competition
   - Improved terrain generation to create a more diverse set of maps 
   - Modularized configuration options to support more types of research 
   - Changed spawning to place agents around the edges of large maps rather than at the center

The v1.5-competition branch contains code and trained models for the experiments in the paper. You can also build the current v1.5 release if you just want to try out the environment and associated tools. Note that there is a GPU-specific and difficult to reproduce issue in a third-party dependency of v1.5. We intend to track down and fix this before the official release. If you encounter this bug, running in CPU mode should suffice to demo the platform.

|icon| Source Material
----------------------

Massively Multiplayer Online role-playing games simulate persistent virtual worlds with hundreds to thousands of concurrent players per server. They are simpler than the real world but arguably much more complex than any game environment studied to date. MMOs support diverse gameplay, intentional specialization across different skills, moderately realistic economies with market forces driven by player interaction, and emergent ad-hoc social structures organized around various in-game activities. Unlike previously studied level- and round-based games, players in MMOs gain intrinsic abilities and tradable items which grant them various advantages that persist throughout hundreds to thousands of hours of play. This sort of long-term reasoning within a dynamic society of other intelligent agents is the default setting for real-world learning but notably absent in previously studied games. Neural MMO lacks the complexity of commercial MMOs developed by large professional studios, but it does adapt several key properties of the genre for computationally efficient research: long time horizons, large populations, and open-ended task specification, among others.

|icon| Achievement System
-------------------------

Neural MMO provides a reward of -1 for dying and 0 otherwise by default. As mentioned in the main text, users may define their own reward functions with full access to game state. We have recently begun experimenting with an achievement system (see figure below) that rewards agents for reaching gameplay milestones. For example, agents may receive a small reward for obtaining their first piece of armor, a medium reward for defeating three other players, and a large reward for traversing the entire map. The tasks and point values themselves are clearly domain-specific, but we believe this achievement system has several advantages compared to traditional reward shaping. First, since each task may be achieved only once, agents cannot simply farm one task repeatedly as sometimes occurs with poorly tuned dense rewards. Second, this property should make the achievement system less sensitive to the exact point tuning. Finally, attaining a high achievement score somewhat guarantees complex behavior since that is exactly what the point system was designed to represent. In practice, we have not yet observed a significant difference compared to training on lifetime alone. We have chosen to include this system in the Supplement nonetheless because it has been successful across a wide genre of human games, and we believe that this is likely to translate into a benefit for training and evaluation in the future.

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

MMOs are computationally efficient compared to most other game genres, meaning that the hardware required to simulate environments is much less than that required to simulate models. This allows us to develop in pure python with all the advantages therein. The only major downside is that computing observations by traversing the relevant python objects is still expensive. In fact, doing this naively is 50-100x slower than the entire rest of the environment. Serialization allows us to perform this computation around 50x faster. Our serialization scheme relies on the observation of each object being represented as a vector. This allows us to maintain a table where each row is an object and each column is an attribute. In practice, we maintain separate tables for each object type as well as for discrete and continuous attributes, but this is a minor implementation detail. The key point is that this representation enables us to compute observations by selecting rows from the table. We wrap all observable object classes with syntactic sugar that updates the table each time an attribute changes. We also maintain a grid of object IDs corresponding to tile and agent positions. Since agents observe a square crop of tiles around them, this grid will always contain the row indices corresponding to nearby objects. If we had chosen to represent maps in continuous space, a similar optimization would be possible using kD trees, but it would be slower by a logarithmic factor.

|icon| Designing for Efficient Complexity
-----------------------------------------

The environments and platforms enumerated in the main text are too diverse to fit cleanly along a single axis of usefulness to research. Qualitative innovations aside, new environments are typically evaluated on the basis of *efficiency* and *complexity*. We sought to define these terms precisely while designing Neural MMO and found that neither are as straightforward as standard usage suggests. This section explains why these terms are more complex than they seem and suggests more precise (albeit still approximate) evaluation criteria.

The REPS Measure of Computational Efficiency
********************************************

Efficiency is often presented as simulation frames per second (fps). This definition is reductive and potentially misleading. The first issue is that environments may exhibit variable performance depending on the number and quality of agents in the simulation. For example, Neural MMO supports configurations that add agents to the environment at a constant rate. As policies improve, agents live longer on average, leading to a higher average population. Simulation speed decreases linearly with the number of agents. In addition, skilled agents may trigger complex environment features, such as harvesting more resources or incurring more combat calculations. This necessarily leads to a performance decrease as well. 

The second issue is data correlation. Not all samples are created equally. For example, simulating the same game environment at a higher frame rate produces more correlated data that, in most circumstances, is likely to be less useful for training than simply simulating twice as many episodes. There is no straightforward known approach to computing semantic sample correlation. The fact that gradient scale tends to increase throughout training suggests that this correlation is probably not even constant within a single environment.

Taking into account the complicating factors outlined above, there is no known method for evaluating environment efficiency in general. However, it is fairly straightforward to improve over the naive fps computation. We propose Real-time Experience Per Second (REPS) as a metric of efficiency -- that is, we evaluate the number of seconds of agent observations collected per second of sampling:

**Real-time experience per second = Independently controlled agent observations / (Simulation time * Real time fps * Cores used)**

We choose to explicitly reference the **number of independently controlled agents** instead of the *total* number of agents because it is possible to treat several agents as a single larger meta-agent by computing their actions jointly. As mentioned above, some environments exhibit variable agent population sizes and simulation speeds depending on the policy. To compute efficiency independently from these factors, we suggest collecting several episodes using the best available policy. Compute average samples per second by dividing the total number of observations collected by the environment **simulation time** used to collect them. This computation should be normalized by the number of **cores used** and ideally standardized to a fixed CPU architecture. Finally, divide by the **real time fps** -- e.g. the frame rate required to run the environment in real time. This term attempts to decorrelate observations by normalizing from total frame count to wall clock time. Since real-time games are designed for human cognition speed, we argue that this heuristic is a better proxy for data independance than total number of frames. For turn-based games, we suggest using the average length of professional matches.

It is common practice to simulate game environments at a lower fps than would be used in human games. Doing so effectively modifies the game as intended and can create problems during training, such as skipping key animations or reducing action precision. However, this approach has demonstrated success on several environments, and, real-time fps should be computed using the downsampled rate for fairness to such works. If an approach employ different frame rates for observations and actions, use the higher of the two.

Difficulty does not Imply Effective Complexity
**********************************************

Environments can be difficult for the wrong reasons. Being easy for humans but hard for modern algorithms is not necessarily "good." For example:

**Observation:** "Type this sentence to win"
**Action Space:** Keyboard input
**Reward:** 1 for inputting "Type this sentence to win" and 0 otherwise

This environment is trivial for humans but nearly impossible for modern reinforcement learning methods because it relies on knowledge external for the environment. From the perspective of an untrained agent, this environment is a deep MDP with a large branching factor and no intermediate rewards. Human priors often grant similar advantages that enable them to quickly solve common reinforcement learning benchmarks such as Atari. This should not be surprising -- games are designed to trigger human priors so that new players can begin playing almost immediately at a basic level of competence. However, humans take much longer to solve graphically randomized games while reinforcement learning agents are agnostic (Dubey et al. Investigating Human Priors for Playing Video Games, https://arxiv.org/pdf/1802.10217.pdf).

We believe it is important to recognize that creating difficulty in this manner is unfair. The toy task above is unsolvable except by using real-world priors, and it is a poor benchmark for reinforcement learning algorithms without them. Most ABI research targeting existing human games handle this disadvantage implicitly using one of two approaches. 

The first approach is to encode human priors into the agent or learning algorithm. Reward shaping, imitation learning on human demonstrations, and exploration bonuses exploiting in-game positional information all fall into this category. In the future, it may also be possible to incorporate pretrained vision and language models, which could allow agents to learn within realistic environments at more comparable speeds to humans. However, it should be noted that this approach places the burden of work upon algorithms researchers rather than upon environment designers. While it is quite likely that methods for incorporating human priors will be important to the long-term advancement of ABI, environments that require them are inapropriate for current algorithms and architecture research.

The second approach is to design environments that require fewer human priors and feature smoother, tutorialized curricula that *teach gradually through experience* (Nicolae Berbece. This is a Talk about Tutorials, Press A to Skip, https://www.youtube.com/watch?v=VM1pV_6IE34&t=1s). This approach is more difficult for designers because environments still need to be interpretable to the researchers using them. In exchange, it unburdens algorithms researchers from having to encode additional priors themselves, which can help avoid difficulties in fairly comparing methods that implicitly assume different human knowledge.

Neither of these approaches is perfect. We have no way of encoding *all* human priors into learning algorithms, and it is also quite difficult to design interpretable environments that require *no* human priors. In either case, scaling up training bridges the gap by brute-forcing bad curricula. Successfully encoding or eliminating human priors -- or even encoding some and eliminating others -- can allow reinforcement learning methods to solve more complex problems at any fixed hardware scale.

Humans Rely Heavily upon Communal Knowledge
*******************************************

Reinforcement learning algorithms are commonly viewed as tremendously sample inefficient compared to human learning. Human players can learn to play new Atari games in only a few minutes while RL agents can require upwards of a billion frames. OpenAI Five was trained by playing 10,000 **years** of DoTA 2; by comparison, professional players average 10,000-25,000 **hours** (https://www.youtube.com/watch?v=saFuWSTXu-w -- Single-digit thousands appears to be the low-end for early-career professionals and 30k is the high-end, or around 1-4 years of continuous play)

There are two problems with this comparison. First, as discussed above, humans do not begin learning games *tabula rasa* -- they have strong priors from both evolution and their own life experience. Second, humans learn from each others' experiences. Most large games have communal wikis, forums, and content creators. Some even have player-run social structures for teaching and mentoring new players. While no one human has played more than a few years of DoTA, the community as a whole has far exceeded 10,000 years. While the impact of the communal knowledge appears intuitively large, it is difficult to quantify exactly. On one hand, only a small fraction of players contribute to communal resources or invent genuinely novel strategies; on the other hand, the strategic landscape of the game has continually evolved since early betas in 2011. Once they learn the basics, new players have immediate access to a decade of strategic refinement produced by an active community of millions.

It may seem odd that we characterize the progression of game-specific knowledge much the same as the progression of scientific knowledge. The recent NetHack wrapper release to support single-agent research is of timely relevance. The game has attracted a small but dedicated following that has continued to innovate new strategies since the original 1987 release. While detailed wikis and resources exist for new players, some choose to figure the game out on their own. This is an important point for evaluation that is not possible in multiplayer games because players will learn strategies from each other regardless of whether they engage with the wikis. NetHack players who do not engage with the wikis report taking as long as 15 years to clear the game for the first time. Some players learn more quickly, but taking several years is quite common. In contrast, players who leverage the wealth of communal knowledge surrounding the game typically clear it within weeks or months: as in the real world, we stand on the shoulders of giants.

Even accounting for all of the above, reinforcement learning algorithms still may be sample inefficient -- we simply argue that the conclusion is not as obvious as it may initially appear and that we should duly consider the environment-external advantages that humans posses. 

Application to Neural MMO
*************************

Our experiments train large agent populations through pure self-play with no human data, a very sparse and general reward function, and a simple architecture with plain PPO. We are able to achieve capable play with a single GPU per experiment in only a few days of training. Most games designed for humans run at 30 frames per second (FPS) at a bare minimum -- 60 FPS is more standard, and 144 FPS is considered an advantage for most high-level competitive games. Simulating these games produces a lot of highly correlated data that is more representative of twitch reactions than of high-level decision making, which typically occurs over longer horizons.

MMOs are a much more efficient choice for the latter: it is an old game genre originally designed to support small cities of players on 90s hardware. Servers typically run at slow internal rates, but client-side animation is used to maintain smooth play. Simulating Neural MMO in real-time requires one state per 0.6 seconds -- less than 2 FPS. Given that samples over the period of a second tend to be highly correlated, this immediately makes the environment one to two orders of magnitude more efficient than many other game genres. We can simulate Neural MMO and all 128 agents upon it on a single CPU core at around 10x real-time speed.

The game mechanics themselves are also designed to *teach gradually through experience*. For example, resources are placed close together at the edges of the map. As agents explore towards the center (especially in larger maps), resources become more spread out. Teaching an agent to solve the latter long-horizon navigation problem from scratch would be difficult, but learning the same end behavior from a curriculum of easier foraging problems is not.
