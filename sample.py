# %%

from mumec import Simulator, World
import matplotlib.pyplot as plt
import numpy as np


class Policy:

    @staticmethod
    def Evaluate(env: Simulator, pie, verbose=0, render=False):

        c = np.zeros((env.M,), dtype=np.float32)
        env.Reset()
        if verbose > 0: print(f'\n[{env.ts}]\n{env.state=}\n{env.costs=}\n{env.weighted_costs=}')
        if render: env.Render()

        al = []
        d = False

        while not d:
            r = env.Step(pie(env.state))
            d = env.terminated or env.truncated
            c += r
            al.append(env.last_action.copy())

            if verbose > 0:print(f'\n{env.last_action}\n[{env.ts}]\n{env.state=}\n{env.costs=}\n{env.weighted_costs=}')
            if render:env.Render()

        if verbose > 0:print(f'\n[{env.ts}]\n{env.state=}\n{env.costs=}\n{env.weighted_costs=}')
        if render: env.Render()

        return c, al

    class Greedy:
        def __call__(self, state):
            return None

    class Random:
        def __init__(self, n_users, n_actions, seed):
            self.rng = np.random.default_rng(seed=seed)
            self.M = int(n_users)
            self.n = int(n_actions)

        def __call__(self, state):
            return self.rng.integers(0, self.n, size=self.M).tolist()

    class Constant:
        def __init__(self, a, n_users):
            self.a = int(a)
            self.M = int(n_users)

        def __call__(self, state):
            return [self.a for _ in range(self.M)]






random_evaluate = False
constant_evaluate = False

greedy_evaluate = True
greedy_verbose, greedy_render = True, False
res = {}

env = Simulator(
    World.New(3, 10),
    weights=[0.5, 0.5],
    partial=False,
    frozen=True,
    seed=None,
)
#-------------------
# Greedy
#-------------------
if greedy_evaluate:
    rets, acts = Policy.Evaluate(
        env,
        Policy.Greedy(),
        verbose=greedy_verbose,
        render=greedy_render,
    )
    res['greedy'] = rets
    print(f'\ngreedy')
    print(acts)
    print(f'cumulative cost per user: {rets}')
    print(f'total cumulative cost: {rets.sum()}')

#-------------------
# Random
#-------------------
if random_evaluate:
    n_random_avg = 3

    for a in range(n_random_avg):
        rets, acts = Policy.Evaluate(
            env,
            Policy.Random(
                n_users=env.M,
                n_actions=env.N+1,
                seed=None,
            ),
        )
        res[f'random-{a}'] = rets
        print(f'\nrandom-{a}')
        print(acts)
        print(f'cumulative cost per user: {rets}')
        print(f'total cumulative cost: {rets.sum()}')


#-------------------
# Constant
#-------------------
if constant_evaluate:
    for a in range(env.N+1):
        rets, acts = Policy.Evaluate(
            env,
            Policy.Constant(a, env.M),
        )
        res[f'constant-{a}'] = rets
        print(f'\nconstant-{a}')
        print(acts)
        print(f'cumulative cost per user: {rets}')
        print(f'total cumulative cost: {rets.sum()}')


#-------------------
# Plot Compare
#-------------------
if res:
    x = list(res.keys())
    user_costs = np.stack([res[k] for k in x], axis=0)
    y = user_costs.sum(axis=1)

    s = np.argsort(y)
    xs = [x[i] for i in s]
    ys = [y[i] for i in s]

    for k, v in zip(xs, ys):
        print(f'{k}: \t{v}')

    plt.figure(figsize=(len(res)*1.5, 6))
    plt.bar(xs, ys)
    plt.title('Policy comparison')
    plt.show()


# %%
