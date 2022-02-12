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
        self.team_size = config.NENT // config.NPOP

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

def spawn_concurrent(config):
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

    return s1 + s2 + s3 + s4
