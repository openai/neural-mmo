[ags]: resource/docs/ags.png?raw=true
[earth]: resource/docs/earth_thumbnail.png
[fire]: resource/docs/fire_thumbnail.png
[water]: resource/docs/water_thumbnail.png
[air]: resource/docs/air_thumbnail.png
[env]: resource/docs/env.jpg
[demo]: v1.2.png

**Status:** Archive (code is provided as-is, no updates expected)

# ![][ags] Neural MMO: A Massively Multiagent Game Environment for Training and Evaluating Intelligent Agents

## (9/19) This is an archive of the inital release of the [Neural MMO project](https://openai.com/blog/neural-mmo/). Active development continuing at [jsuarez5341/neural-mmo](https://github.com/jsuarez5341/neural-mmo). At the time of archiving, the project is on version 1.2 and has made significant progress since v1.0 -- here is a screenshot:

![][demo]

# ![][ags] Neural MMO v1.0 Release

This environment is the first neural MMO; it attempts to create agents that scale to real world complexity. Simulating evolution on Earth is computationally infeasible, but we can construct a reasonable and efficient facsimile. We consider MMORPGs (Massive Multiplayer Online Role Playing Games) the best proxy for the real world among human games: they are complete macrocosms featuring thousands of agents per persistent world, diverse skilling systems, global economies, and ad-hoc high stakes single and team based conflict.

![][env]

## ![][ags] Quickstart

```
# Recommended Setup:
mkdir projekt
cd projekt

git clone https://github.com/jsuarez5341/neural-mmo-client
cd neural-mmo-client
bash setup.sh
cd ..

git clone https://github.com/openai/neural-mmo
cd neural-mmo
bash scripts/setup/setup.sh
python setup.py
```

This will download both the OpenAI environment and the independent client, which is required for rendering. Some setups may require you fix the symlink to the client (e.g. `ln -s ../../neural-mmo-client/ embyr` from `forge/embyr`). The setup file will only install _external_ dependencies -- we assume you have already setup **Anaconda with Python 3.6+**. The environment is framework independently, but our experiment code does depend on PyTorch -- set this up separately.

Run the following, then navigate to http://localhost:8080/forge/embyr/ in Firefox or Chrome to pull up the renderer. Click to start and give it a few seconds to load assets and stabilize fps. For troubleshooting, see the [Client Repo](https://github.com/jsuarez5341/neural-mmo-client).
```
python Forge.py --render #Run the environment with rendering on
```

## ![][ags] Overview

The project is divided into four modules:

| Engineering   | Research      |
| ------------- |-------------|
| ![][earth] Blade: Env | ![][water] Trinity: API   |
| ![][fire] Embyr: Render  | ![][air] Ethyr: Neural |

The objective is to create agents that scale to the complexity and robustness of the real world. This is a variant phrasing of "artificial life." A key perspective of the project is decoupling this statement into subproblems that are concrete, feasible, and directly composable to solve the whole problem. We split the objective into "agents that scale to their environment" and "environments that scale to the real world." These are large respective research and engineering problems, but unlike the original objective, they are specific enough to attempt individually. For a more thorough overview of the project approach and objective, see this [Two Pager](https://docs.google.com/document/d/1_76rYTPtPysSh2_cFFz3Mfso-9VL3_tF5ziaIZ8qmS8/edit?usp=sharing).

![][water]![][air] Research: Agents that scale to env complexity

![][earth]![][fire] Engineering: Env that scales to real world complexity

### ![][water] Trinity
Trinity is the native API for researchers (the naming is simply flavor -- see "Namesake" below). It consists of three base classes, Pantheon, God, and Sword, which you can override to execute code at the Cluster, Server, and Agent levels, respectively.

```python
from forge.trinity import smith, Trinity, Pantheon, God, Sword
trinity = Trinity(Pantheon, God, Sword)
envs = smith.Native(config, args, trinity)
envs.run()
```
That's it -- all communications are handled internally. The provided Pantheon class includes sample gradient aggregation, optimization, and model saving code. The provided God class is mainly a stub to enable future work incorporating population level training algorithms. The provided Sword class contains our simple fully connected model. The full API is defined in forge/trinity/trinity.py.

The VecEnv/Gym computation model not well suited to this setting and performs 10-1000x more interprocess communication than is required. The native API is simpler, more efficient, and requires much less code. However, we do provide a canonical Gym API minimally modified to support a variable number of agents. Forge.py contains a complete forward pass example for a single population. Basic usage is:

```python
from forge.trinity import smith
envs = smith.VecEnv(config, args, self.step)

#The environment is persistent: call reset only upon initialization
obs = envs.reset()

#Observations contain entity and stimulus
#for each agent in each environment.
actions = your_algorithm_here(obs)

#The environment is persistent: "dones" is always None
#If an observation is missing, that agent has died
obs, rewards, dones, infos = envs.step(actions)
```

You can try these both out with:
```
# Run Options:
python Forge.py --nRealm 2 --api native #Run 2 environments with native API
python Forge.py --nRealm 2 --api vecenv #Run 2 environments with vecenv API
```

### ![][air] Ethyr
Ethyr is the "contrib" for this project. It contains useful research tools for interacting with the project. I've seeded it with the helper classes for our experiments, including a model save/load manager, a rollout objects, and a basic optimizer. If you would like to contribute code (in any framework, not just PyTorch), please submit a pull request.

### ![][earth] Blade
Blade is the core environment, including game state and control flow. Researchers should not need to touch this, outside perhaps importing core configurations and enums.
### ![][fire] Embyr
Embyr is the independent THREE.js web client. This is downloaded from a separate repository in the setup above and symlinked to the OpenAI repository. You may have to fix the symbolic link if you get forge/embyr import errors. Again, researchers should not have to edit this. In order to run it, run Forge.py with --render enabled, then navigate to localhost:8080/forge/embyr in Firefox. It will take a couple seconds to initialize and load assets. You will need to refresh the page whenever you reboot the server (Forge.py). Chrome and Safari might work, but we do not currently offer official support.

## ![][ags] Failure Modes
Evaluation can be somewhat difficult in our setting but is not a major blocker. For smaller experiments, we find population size and resource utilization to be reasonable metrics of success. For larger experiments with sufficient domain randomization, Tournaments (as described in the accompanying paper) allow for cross validation of approaches.

We are currently aware of three failure cases for the project:
  * Computational infeasibility
  * "Agents that scale to their environment" is too hard
  * "Environments that scale to the real world" is too hard

The first failure case is a serious risk, but is shared among all areas of the field. This project is not uniquely compute intensive -- in fact, it is one of few environments where it is straightforward to train reasonable policies on a single CPU. If scale is the main issue here, it is likely shared among most if not all other approaches.

The second problem is probably most familiar to researchers as exploration. Given a cold start, how can agents bootstrap both to better policies and to better exploration strategies? This is a hard problem, but it is unlikely to kill the project because:
  * This is independently an important problem that many researchers are already working on already
  * The environment of this project is designed collaboratively to assist agents early on in learning, rather than adversarially as a hard benchmark
  * [Recent](https://blog.openai.com/openai-five/) [projects](https://blog.openai.com/learning-dexterity/) have demonstrated success at scale.

The third problem probably appears most likely to many researchers, but least likely to anyone who has spent a significant amount of time in MMOs. Here is a map of the NYC subway:

![QuestMap](resource/docs/quests.png)
[Source](https://www.reddit.com/user/Gamez_X)

Actually, it's a quest map of Runescape, a particular MMO that our environment is loosely based upon. Each quest is a puzzle in itself, takes anywhere from several minutes to several hours to complete, is part of an interconnected web of prerequisites of other quests, and provides different incentives for completion ranging from equipment to unlockable content to experience in a tightly connected set of skills:

![Skills](resource/docs/skills.png)
![Equipment](resource/docs/equipment.png)
[Source](https://www.jagex.com/en-GB/)

In a massive open world:

![GameMap](resource/docs/map.png)
[Source](https://www.jagex.com/en-GB/)

The most complex class of games considered to date is MOBAs (Massive Online Battle Arenas, e.g. Dota, Quake CTF), which are round based, take on order of an hour, and are mechanically intensive. Achieving 99 in all skills and acquiring the best gear in Runescape takes, at minimum, several thousand hours. In a tournament setting where attacking other players is allowed everywhere, moment-to-moment gameplay is less important than balancing the risks and rewards of any potential strategy--especially in the presence of hundreds of other players attempting to do the same. There is almost certainly still a complexity gap from MMOs to the real world, but we believe it is much smaller than that in environments currently available.

While our environment is nowhere near the level of complexity of a real MMO yet, it does contain key properties of persistence, population scale, and open-endedness. As agents begin to reach the ceiling of the current environment, we plan on continuing development to raise the ceiling.

## ![][ags] File Structure
![][water] **/forge/trinity** ~350 lines
* **/forge/trinity/ann.py** - Defines architectures
* **/forge/trinity/god.py** - Defines server level code (e.g. entity tagging)
* **/forge/trinity/pantheon.py** - Defines cluster level code (e.g. gradient averaging)
* **/forge/trinity/sword.py** - Defines core level code (e.g. running networks, collecting rollouts, computing gradients)
* **/forge/trinity/trinity.py** - Wraps a pantheon, god, and sword
* **/forge/trinity/smith.py** - Defines the Native and VecEnv / Gym APIs

![][air] **/forge/ethyr** ~250 lines
* **/forge/ethyr/rollouts.py** - Collects and merges rollouts
* **/forge/ethyr/stim.py** - Produces a stimulus from local game state
* **/forge/ethyr/torch** - pytorch specific neural utilities
  * **/forge/ethyr/torch/loss.py** - Defines policy/value loss and advantage
  * **/forge/ethyr/torch/optim.py** - Defines optimization and gradient computation
  * **/forge/ethyr/torch/param.py** - Network surgery useful for serialization
  * **/forge/ethyr/torch/stim.py** - Wraps the generic stimulus library with pytorch tensors
  * **/forge/ethyr/torch/utils.py** - Generic pytorch tools

![][earth] **/forge/blade** ~2k lines, of which >1k are for future expansion. Only italicized files are relevant.
  * **/forge/blade/action** - Defines what entities can do, what occurs when they do it, and provides structure for reasoning over actions.
    * **/forge/blade/action/action.py** - Class stubs for each action
    * **/forge/blade/action/tree.py** - Tree structure for assembling actions (e.g. action -> sub-action -> args)
    * **_/forge/blade/action/v2.py_** - Actions that entities can select, instantiate, and .call() to directly modify world state
  * **/forge/blade/core** — Contains ~500 lines of state and game loop code.
    * **_/forge/blade/core/config.py_** - Defines specifications for each test environment, including entity statistics, the map, and spawn locations.
    * **_/forge/blade/core/tile.py_** - Defines an individual game tile
    * **_/forge/blade/core/map.py_** - Defines a map of game tiles
    * **_/forge/blade/core/env.py_** - Wraps the full game state
    * **_/forge/blade/core/realm.py_** - Defines the game loop updating the environment and agents.
  * **/forge/blade/entity** - Defines agents, both neural and scripted
    * **_/forge/blade/entity/player.py_** — Defines "player" state
    * **/forge/blade/entity/npc/**
      * **/forge/blade/entity/npc/npc.py** — This defines client state (e.g. wraps the neural net making decisions)
      * **/forge/blade/entity/npc/mobs.py** - Defines scripted npcs
  * **/forge/blade/hook** - Defines startup scripts that preassemble references
  * **/forge/blade/item** - Defines all items in the game, including equipment
  * **/forge/blade/lib** - Defines generic utilities that either do not have clean python implementations or require specific changes for the project
  * **/forge/blade/systems** - Defines game content

![][fire] **/forge/embyr** See [Client Repo](https://github.com/jsuarez5341/Godsword-Client)

## ![][ags] Namesake
In formal publications, we refer to our project as simply a "Neural MMO." Internally and informally, we call it "Projekt: Godsword." The name comes from two sources: CD Projekt Red, my personal favorite game dev studio, and OldSchool Runescape, which contains an iconic set of weapons called god swords. The latter is a particularly good model for AI environments; the former is more of a soft flavor inspiration.

## ![][ags] Authorship
 I, [Joseph Suarez](https://github.com/jsuarez5341), am the author of the core code base. Yilun Du assisted with running experiments and particularly in setting up tournaments. Phillip Isola and Igor Mordatch have been invaluable collaborators and advisers throughout the project. The client was developed independently as a collaboration between myself and Clare Zhu.

## ![][ags] License & Assets
The OpenAI repository is open sourced under the MIT License. There is a smaller original code base and game kernel that I (Joseph Suarez) retain ownership of, along with associated ideas. I created these before my employment -- the initial commit here represents the latest pre-employment timestep.

Some assets used in this project belong to [Jagex](https://www.jagex.com/en-GB/)
, the creators of Runescape, such as

![Alt text](resource/docs/ags.png?raw=true "Title")
![Alt text](resource/docs/earth_thumbnail.png?raw=true "Title")
![Alt text](resource/docs/water_thumbnail.png?raw=true "Title")
![Alt text](resource/docs/fire_thumbnail.png?raw=true "Title")
![Alt text](resource/docs/air_thumbnail.png?raw=true "Title")

We currently use them for flavor as an homage to the game that inspired the project. We believe these fall under fair use as a not-for-profit project for the advancement of artificial intelligence research -- however, we are more than happy to remove them upon request. We do own the 2D and 3D files for agents.

![Alt text](resource/docs/neuralRED.png?raw=true "Title")
![Alt text](resource/docs/neuralBLUE.png?raw=true "Title")
![Alt text](resource/docs/neuralGREEN.png?raw=true "Title")
![Alt text](resource/docs/neuralFUCHSIA.png?raw=true "Title")
![Alt text](resource/docs/neuralORANGE.png?raw=true "Title")
![Alt text](resource/docs/neuralMINT.png?raw=true "Title")
![Alt text](resource/docs/neuralPURPLE.png?raw=true "Title")
![Alt text](resource/docs/neuralSPRING.png?raw=true "Title")
![Alt text](resource/docs/neuralYELLOW.png?raw=true "Title")
![Alt text](resource/docs/neuralCYAN.png?raw=true "Title")
![Alt text](resource/docs/neuralMAGENTA.png?raw=true "Title")
![Alt text](resource/docs/neuralSKY.png?raw=true "Title")
