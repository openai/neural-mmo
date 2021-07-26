.. |icon| image:: /resource/icon/icon_pixel.png

.. role:: python(code)
    :language: python

.. figure:: /resource/image/splash.png

|icon| Overview
###############

This wiki is devoted to Neural MMO as a game. The current mechanics are fairly straightforward, but there is still some nuance -- especially for those unfamiliar with MMOs. Familiarizing yourself with the game itself will help you understand how to script or train better agents, whether your agents are acting intelligently, and which mistakes suggest potential bugs in your code. While all of the documentation is open to user contributions, that is especially true for this wiki -- simply edit the game_wiki rst and make a pull request. I'll handle rebuilding the docs and testing.

Glossary
********

A quick reference of standard game terms for those unfamiliar:
- **Tick:** The simulation interval of the server; a timestep. With rendering enabled, the server targets 0.6s/tick.
- **NPC:** Non-Player Character; any agent not controlled by a user. Sometimes called a *mob*
- **Spawn:** Entering into the game, e.g. *players spawn with 10 health*
- **RPG:** Role-Playing Game, e.g. a game in which the player takes on a particular role, usually one removed from modern reality, such as that of a knight or wizard. *MMO* is short for *MMORPG*, as most MMOs are also role-playing games.
- **XP (exp):** Experience, a stat associated with progression systems to represent levels.

Agent Actions
*************

Agents may move North, South, East, or West. Submitting a move action is not required, in which case the agent will remain stationary.

If the Combat system (see Game Systems) is enabled, agents may Attack a nearby player or NPC with Melee, Range, and Mage. Submitting an attack is not required, in which case the agent will not attack. Agents cannot damage themselves.

Terrain
*******

Game maps are made of two core materials: Grass and Stone. Players may only walk on stone, and each map is surrounded by a lava border, which kills players on contact. In the current version of the environment, the Resource system adds Forest and Scrub tiles (both of which are traversable), and no other systems add additional tile types.

|icon| Game Systems
###################

While each game system is individually togggleable and configurable, we will assume the cannonical version of the game with all systems enabled and reference interactions between systems. Any such interactions will not be present if that/those game systems are disabled. All numerical values stated below are configurable.

In the base game with no systems enables, players have 10 health and may wander around the map. That's about it.

|icon| Resource System
######################

Adds Food and Water resources. Players spawn with 10 food and 10 water. They lose 1 food and 1 water each tick. Players regenerate 10% health per tick (rounded down) while over 50% food and water.

This system adds a Forest tile, which contains food. Walking over a Forest tile depletes it, replacing it with a Scrub tile and restoring 100% food to the player. Scrub tiles have a 2.5% chance to regenerate into Forest tiles each tick. Walking adjacent to a Water tile restores 100% water to the player.

|icon| Combat
#############

Adds the ability for players to attack other players (and NPCs) with Melee, Range, and Mage combat. These attack styles are usable from 1, 3, and 4 tiles and inflict 7, 3, and 1 damage on a hit respectively. Hit chance is 50% by default. Mage attacks freeze the target in place for 3 ticks by default; this utility and their added range compensates for low damage.

|icon| NPC & Equipment
######################

This system adds NPCs that may be defeated for equipment to the environment.

NPCs are controlled by one of three scripted AIs: Passive, Neutral, and Hostile
 - Passive NPCs wander randomly and cannot attack
 - Neutral NPCs wander randomly but will attack aggressors and give chase using a Dijkstra's algorithm based pathing routine
 - Hostile NPCs will actively hunt down and attack other NPCs and players using the same pathing algorithm

 The exact number and power distribution of NPCs varies by environment config. Generally, Passive NPCs will spawn towards the edges of the map, Hostile NPCs spawn in the middle, and Neutral NPCs spawn somewhere between. The max NPC level (as defined by the Progression system) is 30 for Small maps and 99 for Large maps.

 Upon defeat, NPCs drop equipment for the character that dealt the final blow. Equipment provides a defensive bonus that reduces the accuracy of incoming attacks. Higher level NPCs are likely to have higher level armor, which confers better bonuses.

|icon| Progression
##################

This system allows players to improve at the abilities provided by other game systems by representing them as trainable skills. MMOs typically feature a very slow leveling system spanning hundreds to thousands of hours; in order to make progression relevant to the settings we are currently considering in Neural MMO, we have set a global PROGRESSION_BASE_XP_SCALE multiplier to around 10x that of traditional MMOs.

Fishing
*******

Base Level: 10

XP Gained: base * water restored

Enables players to carry more water, equal to their current level

Hunting
*******

Base Level: 10

XP Gained: base * food restored

Enables players to carry more food, equal to their current level

Constitution
************

Base Level: 10

XP Gained: 2 * base * (damage received + damage dealt)

Increases player health, equal to their current level

Defense
*******

Base Level: 1

XP Gained: 4 * base * damage received

Decreased attacker accuracy. When a player makes an attack, the level of their offensive skill is used as the attack stat. The defense stat is 70% of the target's corresponding attack skill plus 30% of their defense skill. The difference between the attack and defense stats is used to compute a difficulty, which the game rolls against to determine whether the attack hits.

Melee
*****

Base Level: 1

XP Gained: 4 * base * damage dealt

Enables players to inflict more damage with melee according to:

Damage = floor(7 + level * 63 / 99)

Range
*****

Base Level: 1

XP Gained: 4 * base * damage dealt

Enables players to inflict more damage with range according to:

Damage = floor(3 + level * 32 / 99)

Mage
****

Base Level: 1

XP Gained: 4 * base * damage dealt

Enables players to inflict more damage with mage according to:

Damage = floor(1 + level * 24 / 99)
