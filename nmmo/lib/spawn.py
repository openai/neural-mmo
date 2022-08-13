from pdb import set_trace as T
import numpy as np


class SequentialLoader:
    '''config.PLAYER_LOADER that spreads out agent populations'''
    def __init__(self, config):
        items = config.PLAYERS
        for idx, itm in enumerate(items):
           itm.policyID = idx 

        self.items = items
        self.idx   = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx = (self.idx + 1) % len(self.items)
        return self.idx, self.items[self.idx]

class TeamLoader:
    '''config.PLAYER_LOADER that loads agent populations adjacent'''
    def __init__(self, config):
        items = config.PLAYERS
        self.team_size = config.PLAYER_N // len(items)

        for idx, itm in enumerate(items):
           itm.policyID = idx 

        self.items = items
        self.idx   = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += 1
        team_idx  = self.idx // self.team_size
        return team_idx, self.items[team_idx]


def spawn_continuous(config):
    '''Generates spawn positions for new agents

    Randomly selects spawn positions around
    the borders of the square game map

    Returns:
        tuple(int, int):

    position:
        The position (row, col) to spawn the given agent
    '''
    #Spawn at edges
    mmax = config.MAP_CENTER + config.MAP_BORDER
    mmin = config.MAP_BORDER

    var  = np.random.randint(mmin, mmax)
    fixed = np.random.choice([mmin, mmax])
    r, c = int(var), int(fixed)
    if np.random.rand() > 0.5:
        r, c = c, r 
    return (r, c)

def old_spawn_concurrent(config):
    '''Generates spawn positions for new agents

    Evenly spaces agents around the borders
    of the square game map

    Returns:
        tuple(int, int):

    position:
        The position (row, col) to spawn the given agent
    '''
 
    left   = config.MAP_BORDER
    right  = config.MAP_CENTER + config.MAP_BORDER
    rrange = np.arange(left+2, right, 4).tolist()

    assert not config.MAP_CENTER % 4
    per_side = config.MAP_CENTER // 4
  
    lows   = (left+np.zeros(per_side, dtype=np.int)).tolist()
    highs  = (right+np.zeros(per_side, dtype=np.int)).tolist()

    s1     = list(zip(rrange, lows))
    s2     = list(zip(lows, rrange))
    s3     = list(zip(rrange, highs))
    s4     = list(zip(highs, rrange))

    ret = s1 + s2 + s3 + s4

    # Shuffle needs porting to competition version
    np.random.shuffle(ret)

    return ret

def spawn_concurrent(config):
    '''Generates spawn positions for new agents

    Evenly spaces agents around the borders
    of the square game map

    Returns:
        tuple(int, int):

    position:
        The position (row, col) to spawn the given agent
    '''
    team_size = config.PLAYER_TEAM_SIZE
    team_n = len(config.PLAYERS)
    teammate_sep = config.PLAYER_SPAWN_TEAMMATE_DISTANCE

    # Number of total border tiles
    total_tiles = 4 * config.MAP_CENTER 

    # Number of tiles, including within-team sep, occupied by each team
    tiles_per_team = teammate_sep*(team_size-1) + team_size

    # Number of total tiles dedicated to separating teams
    buffer_tiles = 0
    if team_n > 1:
        buffer_tiles = total_tiles - tiles_per_team*team_n

    # Number of tiles between teams
    team_sep = buffer_tiles // team_n

    # Accounts for lava borders in coord calcs
    left = config.MAP_BORDER
    right = config.MAP_CENTER + config.MAP_BORDER
    lows = config.MAP_CENTER * [left]
    highs = config.MAP_CENTER * [right]
    inc = list(range(config.MAP_BORDER, config.MAP_CENTER+config.MAP_BORDER))

    # All edge tiles in order
    sides = []
    sides += list(zip(lows, inc))
    sides += list(zip(inc, highs))
    sides += list(zip(highs, inc[::-1]))
    sides += list(zip(inc[::-1], lows))

    # Space across and within teams
    spawn_positions = []
    for idx in range(team_sep//2, len(sides), tiles_per_team+team_sep):
        for offset in list(range(0,  tiles_per_team, teammate_sep+1)):
            if len(spawn_positions) >= config.PLAYER_N:
                continue

            pos = sides[idx + offset]
            spawn_positions.append(pos)

    return spawn_positions

