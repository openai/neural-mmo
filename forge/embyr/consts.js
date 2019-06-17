const tick = 0.6;
const nAnim = 16;
const worldWidth = 64, worldDepth = 64;
const worldHalfWidth = worldWidth / 2;
const worldHalfDepth = worldDepth / 2;
const sz = 64;
const tileSz = 64;
const nTiles = 80;

const taiyaki_sky = 0x4477aa;
const seal_sky = 0x003333;

const modes = { ADMIN: 0, PLAYER: 1, SPECTATOR: 2};
const views = { CLIENT: 0, COUNTS: 1, VALUES: 2 };

const tiles = {
    0: "lava",
    1: "water",
    2: "grass",
    3: "scrub",
    4: "forest",
    5: "stone",
    6: "orerock",
}
const Neon = {
   'RED':      '#ff0000',
   'ORANGE':   '#ff8000',
   'YELLOW':   '#ffff00',

   'GREEN':    '#00ff00',
   'MINT':     '#00ff80',
   'CYAN':     '#00ffff',

   'BLUE':     '#0000ff',
   'PURPLE':   '#8000ff',
   'MAGENTA':  '#ff00ff',

   'FUCHSIA':  '#ff0080',
   'SPRING':   '#80ff80',
   'SKY':      '#0080ff',

   'WHITE':    '#ffffff',
   'GRAY':     '#666666',
   'BLACK':    '#000000',

   'BLOOD':    '#bb0000',
   'BROWN':    '#7a3402',
   'GOLD':     '#eec600',
   'SILVER':   '#b8b8b8',

   'TERM':     '#41ff00',
   'MASK':     '#d67fff'
}

const tileHeights = {
   0: 1,
   1: 0,
   2: 1,
   3: 1,
   4: 1,
   5: 2,
   6: 1
}
