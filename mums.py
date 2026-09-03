
import math, json
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiDiscrete, Dict as DictSpace


def JLoad(path):
    with open(path, 'r') as f: obj = json.load(f)
    return obj

def JSave(path, obj, indent=4):
    with open(path, 'w') as f: json.dump(obj, f, indent=indent)

class Gene:
    def __init__(self, comp_size_range, data_size_range, zero_prob, seed=None):
        self.csize_low, self.csize_high =  comp_size_range
        self.dsize_low, self.dsize_high =  data_size_range
        self.zero_prob = zero_prob
        self.zero_task = (0.0, 0.0)
        self.seed = seed
        self.reset()

    def reset(self): self.rng = np.random.default_rng(self.seed)

    def generate(self):
        if self.rng.random() < self.zero_prob: return self.zero_task
        else: return (
            self.rng.uniform(self.csize_low, self.csize_high),
            self.rng.uniform(self.dsize_low, self.dsize_high),
        )

class Mob:
    def __init__(self, trajectory):
        if not trajectory: raise ValueError("trajectory must not be empty")
        trajectory =  trajectory + list(reversed(trajectory[:-1])) 
        self.trajectory = [tuple(map(float, p)) for p in trajectory]
        self.count = len(self.trajectory)
        self.reset()

    def reset(self): self.i = 0
    def generate(self):
        point = self.trajectory[self.i]
        self.i = (self.i + 1) % self.count
        return point

    @staticmethod
    def FromWayPoints(waypoints: list[tuple[float, float]], speed: float, max_steps=None):
        if len(waypoints) < 2: raise ValueError("Need at least two waypoints")
        if speed <= 0.0: raise ValueError("speed must be positive")
        trajectory = [tuple(map(float, waypoints[0]))]
        n_steps = 0
        for i in range(len(waypoints) - 1):
            x0, y0 = waypoints[i]
            x1, y1 = waypoints[i + 1]
            distance = math.hypot(x1 - x0, y1 - y0)
            n = max(1, math.ceil(distance / speed))
             
            for j in range(1, n+1):
                t = j / n
                trajectory.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
                n_steps += 1
                if max_steps and (max_steps == n_steps):  return __class__(trajectory)
                    
        return __class__(trajectory)

    @staticmethod
    def FromRandomWayPoints(n, xrange, yrange, speed, max_steps=None, seed=None):
        rng = np.random.default_rng(seed) 
        return __class__.FromWayPoints([ (rng.uniform(*xrange), rng.uniform(*yrange)) for _ in range(n) ], speed, max_steps)

class UFields:
    x = 0
    y = 1
    exrate = 2
    tsize = 3
    dsize = 4
    txbw = 5
    DIM = 6

class SFields:
    x = 0
    y = 1
    exrate = 2
    ncon = 3
    DIM = 4

class MECWorld(Env):
    """MEC environment with an allocation-simplex action semantics."""

    metadata = {"render_modes": ["human"]}
    FLOAT_DTYPE = np.float32
    INT_DTYPE = np.int64
    DEFAULT_NORMARGS = dict(
        station_cpu_ref=1.0,
        user_cpu_ref=1.0,
        task_size_ref=1.0,
        data_size_ref=1.0,
        bandwidth_ref=1.0,
        tx_rate_ref=1.0,
        )

    @staticmethod
    def TXRATE(
        bw: float,  # allocated bw in Hz
        dis: float, # distance in m
        # constants
        p = 0.45, # tx power in Watts (Joules/Sec)
        c = 3e8, # speed of light m/s
        f = 3.5e9, # carrier frequency Hz
        u = 2e-20, # noise density ~ 10.0 ** ((-174.0 + 7.0) / 10.0) / 1000.0
        eps = 1.0, # min distance if distance is zero
        ) -> float:
        if bw <= 0.0: return 0.0
        path_loss = (4.0 * math.pi * max(dis, eps) / (c / f)) ** 2
        snr = (p / path_loss) / (u * bw)
        return float(bw * math.log2(1.0 + snr))

    def freeze(self, greset=True, mreset=True):  self.greset, self.mreset = bool(greset), bool(mreset)

    def __init__(
        self,
        discrete,
        x_size,
        y_size,
        n_stations,
        n_users,
        n_conns,
        horizon,
        logbase,
        args_stations,
        args_users,
        genes,
        mobs,
        greset=False,
        mreset=False,
        **normargs,
    ):
        super().__init__()
        self.discrete = bool(discrete)
        self.x_size = float(x_size)
        self.y_size = float(y_size)
        self.x_low, self.x_high = -self.x_size / 2.0, self.x_size / 2.0
        self.y_low, self.y_high = -self.y_size / 2.0, self.y_size / 2.0
        self.n_stations = int(n_stations)
        self.n_users = int(n_users)
        self.n_conns = int(n_conns)
        self.horizon = int(horizon)
        assert not (self.n_stations <= 0 or self.n_users <= 0 or not (1 <= self.n_conns <= self.n_stations)), "Invalid environment dimensions"
        assert self.horizon > 0, "horizon must be positive"
        assert not (len(args_stations) != self.n_stations or len(args_users) != self.n_users), "Argument lengths do not match n_stations/n_users"
        assert not(len(genes) != self.n_users or len(mobs) != self.n_users), "genes/mobs lengths must match n_users"
        assert logbase>=2
        self.genes = list(genes)
        self.mobs = list(mobs)
        self.args_stations = list(args_stations)
        self.args_users = list(args_users)
        self.greset = bool(greset)
        self.mreset = bool(mreset)
        self.logmul = np.log(logbase)
        self.norm = normargs if normargs else self.DEFAULT_NORMARGS
        self.station_norm = np.array([
            self.x_size / 2.0,
            self.y_size / 2.0,
            self.norm['station_cpu_ref'],
            1.0
        ], dtype=self.FLOAT_DTYPE)

        self.user_norm = np.array([
            self.x_size / 2.0,
            self.y_size / 2.0,
            self.norm['user_cpu_ref'],
            self.norm['task_size_ref'],
            self.norm['data_size_ref'],
            self.norm['bandwidth_ref']
        ], dtype=self.FLOAT_DTYPE)
        
        self.station_vec = np.zeros((self.n_stations, SFields.DIM), dtype=self.FLOAT_DTYPE)
        self.user_vec = np.zeros((self.n_users, UFields.DIM), dtype=self.FLOAT_DTYPE)
        self.distances = np.zeros((self.n_users, self.n_stations), dtype=self.FLOAT_DTYPE)
        self.associations = np.zeros((self.n_users, self.n_conns), dtype=self.INT_DTYPE)
        self.association_matrix = np.zeros((self.n_users, self.n_stations), dtype=self.FLOAT_DTYPE)
        self.txrates = np.zeros((self.n_users, self.n_conns), dtype=self.FLOAT_DTYPE)
        self._fill_fixed_fields()

        spaces = {
            "stations": Box(-np.inf, np.inf, shape=(self.n_stations, SFields.DIM), dtype=self.FLOAT_DTYPE),
            "users": Box(-np.inf, np.inf, shape=(self.n_users, UFields.DIM), dtype=self.FLOAT_DTYPE),
            "association": Box(-np.inf, np.inf, shape=(self.n_users, self.n_stations), dtype=self.FLOAT_DTYPE),
        }
        self.observation_space = DictSpace(spaces)
        self._make_action_space(self.discrete)
        self.drag_history = np.zeros((self.n_stations,2), dtype=self.FLOAT_DTYPE) # ex_decay, tx_decay

    def _make_action_space(self, discrete):
        if discrete:
            # where to offloaded from all n_conns
            self.action_space = MultiDiscrete(
                nvec=[(self.n_conns+1) for ui in range(self.n_users)], 
                dtype=self.INT_DTYPE)
        else:
            # how much to offload to each n_conns (ratio)
            self.action_space = Box(
                low=0.0, high=1.0,
                shape=(self.n_users, self.n_conns + 1),
                dtype=self.FLOAT_DTYPE,
            )


    def _get_obs(self): return {
            "stations": (self.stations / self.station_norm),
            "users": (self.users / self.user_norm),
            "association": (self.association_matrix / self.norm['tx_rate_ref']),
        }

    def _fill_fixed_fields(self):
        for si, (x, y, exrate) in enumerate(self.args_stations): self.station_vec[si] = (x, y, exrate, 0.0)
        for ui, (exrate, txbw) in enumerate(self.args_users):
            self.user_vec[ui, UFields.exrate], self.user_vec[ui, UFields.txbw] = exrate, txbw

    def reset_genes(self):
        for gene in (self.genes): gene.reset()

    def reset_mobs(self):
        for mob in (self.mobs): mob.reset()

    def update_user_pos(self):
        for ui in range(self.n_users): self.users[ui, UFields.x], self.users[ui, UFields.y] = self.mobs[ui].generate()

    def update_distances(self): self.distances[:] = np.linalg.norm(self.users[:, [UFields.x, UFields.y]][:, None, :] - self.stations[:, [SFields.x, SFields.y]][None, :, :], axis=-1)

    def update_associations(self):

        self.stations[:, SFields.ncon] = 0.0
        self.association_matrix.fill(0.0)

        #bw = self.users[:, UFields.txbw] #/ self.n_conns
        self.txrates.fill(0.0)

        for ui in range(self.n_users):
            nearest = np.argsort(self.distances[ui])[:self.n_conns]
            self.associations[ui] = nearest
            for ci, si in enumerate(nearest):
                self.stations[si, SFields.ncon] += ( 1.0 / self.n_users )
                self.txrates[ui, ci] = __class__.TXRATE( self.users[ui, UFields.txbw].item(), self.distances[ui, si].item())
                self.association_matrix[ui, si] = self.txrates[ui, ci]


    def generate_tasks(self):
        for ui, gene in enumerate(self.genes):
            self.users[ui, UFields.tsize], self.users[ui, UFields.dsize] = gene.generate()

    def _compute_cost(self, action):
        # action = [0 ,1 ,2, 3]

        # keep track of how much total data was offloaded to each server
        offloaded = np.zeros((self.n_stations,2), dtype=self.FLOAT_DTYPE)

        station_rate_decay =  np.log(self.stations[:, SFields.ncon] + 1.0) / self.logmul
        user_latencies = np.zeros(self.n_users, dtype=self.FLOAT_DTYPE)

        #self.drag_history # n_station, 2


        if self.discrete:
            #--------------------------------------------- loop
            for ui in range(self.n_users):
                data_total = self.users[ui, UFields.dsize]
                task_total = self.users[ui, UFields.tsize]
                ci = action[ui]
                if ci==self.n_conns: # no-offload
                    tx_time = 0.0 #(data_total)/np.inf
                    ex_time = (task_total) / self.users[ui, UFields.exrate]
                else:
                    si = int(self.associations[ui, ci])
                    #print(self.txrates[ui, ci],  self.drag_history[si, 1])
                    tx_time = (data_total) / (self.txrates[ui, ci] - self.txrates[ui, ci]*(self.drag_history[si, 1]))
                    ex_time = (task_total) / (self.stations[si, SFields.exrate] * station_rate_decay[si] - self.drag_history[si, 0])
                    offloaded[si, :] += (task_total, data_total) #<--- 

                user_latencies[ui] = tx_time + ex_time
            #--------------------------------------------- loop
        else:
            #--------------------------------------------- loop
            for ui in range(self.n_users):
                data_total = self.users[ui, UFields.dsize]
                task_total = self.users[ui, UFields.tsize]
                times = []
                for ci in range(self.n_conns):
                    si = int(self.associations[ui, ci])
                    #print(action.shape, action)
                    remote_ratio = action[ui, ci]
                    tx_time = (data_total * remote_ratio) / (self.txrates[ui, ci] - self.txrates[ui, ci]*(self.drag_history[si, 1]))
                    ex_time = (task_total * remote_ratio) / (self.stations[si, SFields.exrate] * station_rate_decay[si] - self.drag_history[si, 0])
                    offloaded[si, :] += ((task_total * remote_ratio), (data_total * remote_ratio)) #<--- 
                    times.append(tx_time + ex_time)
                # local action at the end
                times.append((task_total * action[ui, self.n_conns]) / self.users[ui, UFields.exrate])

                user_latencies[ui] = max(times)
            #--------------------------------------------- loop

        self.drag_history[:, 0] = offloaded[:, 0] / (self.norm['task_size_ref']*self.n_users)
        self.drag_history[:, 1] = offloaded[:, 1] / (self.norm['data_size_ref']*self.n_users)
        return user_latencies


    def reset(self, *, seed=None, options=None):
        if self.greset: self.reset_genes()
        if self.mreset:  self.reset_mobs()
        self.users = np.copy(self.user_vec)
        self.stations = np.copy(self.station_vec)
        self.update_user_pos()
        self.update_distances()
        self.update_associations()

        self.generate_tasks()
        self.ts = 0
        self.done = False
        return self._get_obs(), dict(ts=self.ts)

    def step(self, action):
        #allocation = np.clip(action, 0.0, 1.0) 
        # assume actions are normalized
        user_cost = self._compute_cost(action)
        cost = user_cost.mean() 

        self.update_user_pos()
        self.update_distances()
        self.update_associations()

        self.generate_tasks()
        self.ts += 1
        self.done = self.ts >= self.horizon
        return self._get_obs(), -float(cost), self.done, self.done,  dict(ts=self.ts)

    def render(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        station_x, station_y = self.stations[:, SFields.x], self.stations[:, SFields.y]
        user_x, user_y = self.users[:, UFields.x], self.users[:, UFields.y]
        ax.scatter(station_x, station_y, marker="^", s=180, label="MEC Station", zorder=3)
        ax.scatter(user_x, user_y, marker="o", s=100, label="User", zorder=4)
        for si, (x, y) in enumerate(zip(station_x, station_y)):
            ax.annotate(
                f"S{si} ({self.stations[si, SFields.ncon]:.2f}%)",
                (x, y), xytext=(7, 7), textcoords="offset points", fontsize=8,
            )
        for ui, (x, y) in enumerate(zip(user_x, user_y)):
            ax.annotate(
                f"U{ui}",# ({self.users[ui, UFields.tsize]/self.norm['task_size_ref']:.2f}c, {self.users[ui, UFields.dsize]/self.norm['data_size_ref']:.2f}b)",
                (x, y), xytext=(7, -30), textcoords="offset points", fontsize=8,
            )
            for si in self.associations[ui]:
                ax.plot([x, self.stations[si, SFields.x]], [y, self.stations[si, SFields.y]], "--", alpha=0.5, linewidth=1.0)
        ax.set_xlim(self.x_low, self.x_high)
        ax.set_ylim(self.y_low, self.y_high)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"MEC Environment — step {self.ts}/{self.horizon}")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="box")
        ax.legend()
        plt.tight_layout()
        plt.show()
        return 

class DiscreteBaselines:

    def __call__(self, policy, episodes, gamma=1.0, render=False, verbose=False):
        returns = []
        rewards = []
        self.__pie = getattr(self, policy)
        for ep in range(episodes):
            obs, _ = self.env.reset()
            if verbose: print(f'{self.env.ts}\n\t{obs=}')
            if render:  self.env.render()
            total = 0.0
            rewards_ep = []


            while True:
                action = self.__pie()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                rewards_ep.append(reward)
                if verbose: print(f'{self.env.ts}\n\t{obs=}\n\t{action=}\n\t{reward=}\n\t{terminated=}:{truncated=}')
                if render:  self.env.render()
                if terminated or truncated: break

            returns.append(total)
            rewards.append(total / self.env.horizon)

            g = 0.0
            mc = []
            for reward in reversed(rewards_ep):
                g = reward + gamma*g
                mc.append(g)
            mc.reverse()
        self.__pie = None
        return {
            "returns": np.asarray(returns),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_reward": float(np.mean(rewards)),
        }

    def __init__(self, env, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        self.env = env
        self.__pie = None
        self.__pies = [k for k in self.__class__.__dict__ if not k.startswith("__")]

    def __iter__(self): return iter(self.__pies)

    def RandomMixedOffloading(self): return [self.rng.integers(0, self.env.n_conns+1) for _ in range (self.env.n_users)]

    def FullRandomOffloading(self): return [self.rng.integers(0, self.env.n_conns) for _ in range (self.env.n_users)]

    def BestOffloading(self): return [0 for _ in range (self.env.n_users)]

    def NoOffloading(self): return [self.env.n_conns for _ in range (self.env.n_users)] 

class ContinuousBaselines:

    def __call__(self, policy, episodes, gamma=1.0, render=False, verbose=False):
        returns = []
        rewards = []
        self.__pie = getattr(self, policy)
        for ep in range(episodes):
            obs, _ = self.env.reset()
            if verbose: print(f'{self.env.ts}\n\t{obs=}')
            if render:  self.env.render()
            total = 0.0
            rewards_ep = []


            while True:
                action = self.__pie()
                obs, reward, terminated, truncated, _ = self.env.step(action)
                total += reward
                rewards_ep.append(reward)
                if verbose: print(f'{self.env.ts}\n\t{obs=}\n\t{action=}\n\t{reward=}\n\t{terminated=}:{truncated=}')
                if render:  self.env.render()
                if terminated or truncated: break

            returns.append(total)
            rewards.append(total / self.env.horizon)

            g = 0.0
            mc = []
            for reward in reversed(rewards_ep):
                g = reward + gamma*g
                mc.append(g)
            mc.reverse()
        self.__pie = None
        return {
            "returns": np.asarray(returns),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_reward": float(np.mean(rewards)),
        }

    def __init__(self, env, seed=None):
        self.rng = np.random.default_rng(seed=seed)
        self.env = env
        self.__pie = None
        self.__pies = [k for k in self.__class__.__dict__ if not k.startswith("__")]

    def __iter__(self): return iter(self.__pies)

    def RandomOffloading(self): return self.rng.dirichlet(np.ones(self.env.n_conns + 1), size=self.env.n_users).astype(self.env.FLOAT_DTYPE)

    def NoOffloading(self):
        action = np.zeros(self.env.action_space.shape, dtype=self.env.FLOAT_DTYPE)
        action[:, self.env.n_conns] = 1.0
        return action

    def UniformMixedOffloading(self): return np.full(self.env.action_space.shape, 1.0 / (self.env.n_conns + 1), dtype=self.env.FLOAT_DTYPE)
    
    def UniformRemoteOffloading(self):
        action = np.zeros(self.env.action_space.shape, dtype=self.env.FLOAT_DTYPE)
        action[:, :self.env.n_conns] = 1.0 / self.env.n_conns
        return action

    def GreedySingleFullOffloading(self):
        # offloads full task to the best transmission rate 
        action = np.zeros(self.env.action_space.shape, dtype=self.env.FLOAT_DTYPE)
        best = np.argmax(self.env.txrates, axis=1)
        action[np.arange(self.env.n_users), best] = 1.0
        return action

    def NearestSingleFullOffloading(self):
        action = np.zeros(self.env.action_space.shape, dtype=self.env.FLOAT_DTYPE)
        action[:, 0] = 1.0
        return action

    def GreedyOffloading(self):
        action = np.zeros(self.env.action_space.shape, dtype=self.env.FLOAT_DTYPE)
        station_active = np.zeros(self.env.n_stations, dtype=np.int64)
        for ui in range(self.env.n_users):
            for ci in range(self.env.n_conns):
                if self.env.txrates[ui, ci] > 0:
                    station_active[self.env.associations[ui, ci]] += 1

        for ui in range(self.env.n_users):
            task = float(self.env.users[ui, 3])
            data = float(self.env.users[ui, 4])
            local_time = task / float(self.env.users[ui, 2])
            best_ci = 0
            best_time = np.inf
            for ci in range(self.env.n_conns):
                si = int(self.env.associations[ui, ci])
                remote_time = data / float(self.env.txrates[ui, ci]) + task / (
                    float(self.env.stations[si, 2]) / max(1, int(station_active[si]))
                )
                if remote_time < best_time:
                    best_time = remote_time
                    best_ci = ci
            if local_time <= best_time:
                action[ui, self.env.n_conns] = 1.0
            else:
                action[ui, best_ci] = 1.0
        return action

class EnvMaker:

    @staticmethod
    def SampleConfig(): return {
        "description": "Sample Configuration",
        "x_size": 5000.0,
        "y_size": 5000.0,
        "n_stations": 4,
        "n_users": 12,
        "n_conns": 2,
        "horizon": 30,
        "log_base": 3,
        "station_exrate": 12.0e9,
        "user_exrate": 4.0e9,
        "user_txbw": 20.0e6,
        "task_size_range": (1.0e9, 4.0e9),
        "data_size_range": (1.0e6, 8.0e6),
        "task_zero_prob": 0.02,
        "waypoint_count": 6,
        "mobility_speed": 75.0,
        "norm": {
            "station_cpu_ref": 12.0e9,
            "user_cpu_ref": 4.0e9,
            "task_size_ref": 4.0e9,
            "data_size_ref": 8.0e6,
            "bandwidth_ref": 20.0e6,
            "tx_rate_ref": 100.0e6,
        }}

    @staticmethod
    def RandomSeed(rng): return rng.integers(0, 10000000)

    @staticmethod
    def grid_points(n, x_size, y_size):
        nx = int(np.ceil(np.sqrt(n)))
        ny = int(np.ceil(n / nx))
        xs = np.linspace(-x_size / 2, x_size / 2, nx + 2)[1:-1]
        ys = np.linspace(-y_size / 2, y_size / 2, ny + 2)[1:-1]
        return [(float(x), float(y)) for y in ys for x in xs][:n]

    @staticmethod
    def make_genes(n_users, task_size_range, data_size_range, zero_prob, seed):
        rng = np.random.default_rng(seed=seed)
        return [Gene( task_size_range,  data_size_range, zero_prob,  seed=__class__.RandomSeed(rng),) for _ in range(n_users)]

    @staticmethod
    def make_mobility(n_users, x_size, y_size, waypoint_count, speed, max_steps, seed):
        rng = np.random.default_rng(seed=seed)
        return [Mob.FromRandomWayPoints(  waypoint_count, (-x_size / 2, x_size / 2), (-y_size / 2, y_size / 2), speed, max_steps=max_steps,seed=__class__.RandomSeed(rng), ) for _ in range(n_users)]

    @staticmethod
    def Make(discrete, cfg:dict, freeze=False, seed=None):
        rng = np.random.default_rng(seed=seed)
        stations = [
            (x, y, cfg["station_exrate"])
            for x, y in __class__.grid_points(cfg["n_stations"], cfg["x_size"], cfg["y_size"])
        ]
        users = [
            (cfg["user_exrate"], cfg["user_txbw"])
            for _ in range(cfg["n_users"])
        ]
        genes = __class__.make_genes(
            cfg["n_users"],
            cfg["task_size_range"],
            cfg["data_size_range"],
            cfg["task_zero_prob"],
            __class__.RandomSeed(rng),
        )
        mobs = __class__.make_mobility(
            cfg["n_users"],
            cfg["x_size"],
            cfg["y_size"],
            cfg["waypoint_count"],
            cfg["mobility_speed"],
            cfg["horizon"],
            __class__.RandomSeed(rng),
        )
        return MECWorld(
            discrete=discrete,
            x_size=cfg["x_size"],
            y_size=cfg["y_size"],
            n_stations=cfg["n_stations"],
            n_users=cfg["n_users"],
            n_conns=cfg["n_conns"],
            horizon=cfg["horizon"],
            logbase=cfg['log_base'],
            args_stations=stations,
            args_users=users,
            genes=genes,
            mobs=mobs,
            greset=freeze,
            mreset=freeze,
            **cfg["norm"],
        )

    @staticmethod
    def MakeFromFile(discrete, json_file:str, freeze=False, seed=None):
        return __class__.Make(discrete, JLoad(json_file), freeze=freeze, seed=seed)

    @staticmethod
    def MakeFromString(discrete, json_string:str, freeze=False, seed=None):
        return __class__.Make(discrete, json.loads(json_string), freeze=freeze, seed=seed)

    @staticmethod
    def MakeFromDataBase(discrete, name:str, freeze=False, seed=None):
        return __class__.MakeFromString(discrete, __class__.WorldDataBase[name], freeze=freeze, seed=seed)

    @staticmethod
    def ListDataBase(): return list(__class__.WorldDataBase.keys())
    # ---------------------------------------------
    WorldDataBase = dict(
    # ---------------------------------------------

    world1 = """
    {
        "description": "5 km balanced world",
        "x_size": 5000.0,
        "y_size": 5000.0,
        "n_stations": 4,
        "n_users": 12,
        "n_conns": 2,
        "horizon": 30,
        "log_base": 3,
        "station_exrate": 12000000000.0,
        "user_exrate": 4000000000.0,
        "user_txbw": 19000000.0,
        "task_size_range": [
            1000000000.0,
            4000000000.0
        ],
        "data_size_range": [
            1000000.0,
            8000000.0
        ],
        "task_zero_prob": 0.02,
        "waypoint_count": 6,
        "mobility_speed": 75.0,
        "norm": {
            "station_cpu_ref": 12000000000.0,
            "user_cpu_ref": 4000000000.0,
            "task_size_ref": 4000000000.0,
            "data_size_ref": 8000000.0,
            "bandwidth_ref": 20000000.0,
            "tx_rate_ref": 100000000.0
        }
    }
    """,

    world2 = """
    {
        "description": "7.5 km larger world with more users",
        "x_size": 7500.0,
        "y_size": 7500.0,
        "n_stations": 6,
        "n_users": 24,
        "n_conns": 2,
        "horizon": 30,
        "log_base": 4,
        "station_exrate": 16000000000.0,
        "user_exrate":      3000000000.0,
        "user_txbw": 23000000.0,
        "task_size_range": [
            1500000000.0,
            5000000000.0
        ],
        "data_size_range": [
            2000000.0,
            12000000.0
        ],
        "task_zero_prob": 0.02,
        "waypoint_count": 7,
        "mobility_speed": 90.0,
        "norm": {
            "station_cpu_ref": 16000000000.0,
            "user_cpu_ref": 5000000000.0,
            "task_size_ref": 5000000000.0,
            "data_size_ref": 12000000.0,
            "bandwidth_ref": 25000000.0,
            "tx_rate_ref": 100000000.0
        }
    }
    """,

    world3 = """
    {
        "description": "10 km high-load world",
        "x_size": 10000.0,
        "y_size": 10000.0,
        "n_stations": 8,
        "n_users": 40,
        "n_conns": 3,
        "horizon": 40,
        "log_base": 6,
        "station_exrate":   30000000000.0,
        "user_exrate":      4700000000.0,
        "user_txbw": 32000000.0,
        "task_size_range": [
            2000000000.0,
            7000000000.0
        ],
        "data_size_range": [
            2000000.0,
            16000000.0
        ],
        "task_zero_prob": 0.01,
        "waypoint_count": 8,
        "mobility_speed": 110.0,
        "norm": {
            "station_cpu_ref": 20000000000.0,
            "user_cpu_ref": 6000000000.0,
            "task_size_ref": 7000000000.0,
            "data_size_ref": 16000000.0,
            "bandwidth_ref": 30000000.0,
            "tx_rate_ref": 100000000.0
        }
    }
    """,

    world4 = """
    {
        "description": "15 km sparse high-mobility world",
        "x_size": 15000.0,
        "y_size": 15000.0,
        "n_stations": 12,
        "n_users": 60,
        "n_conns": 3,
        "horizon": 40,
        "log_base": 8,
        "station_exrate": 36000000000.0,
        "user_exrate": 3500000000.0,
        "user_txbw": 36000000.0,
        "task_size_range": [
            2500000000.0,
            9000000000.0
        ],
        "data_size_range": [
            3000000.0,
            20000000.0
        ],
        "task_zero_prob": 0.01,
        "waypoint_count": 10,
        "mobility_speed": 140.0,
        "norm": {
            "station_cpu_ref": 24000000000.0,
            "user_cpu_ref": 7000000000.0,
            "task_size_ref": 9000000000.0,
            "data_size_ref": 20000000.0,
            "bandwidth_ref": 40000000.0,
            "tx_rate_ref": 100000000.0
        }
    }
    """,

    # ---------------------------------------------
    )
    # ---------------------------------------------


