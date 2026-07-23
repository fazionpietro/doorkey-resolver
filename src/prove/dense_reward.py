import csv, ast, sys

INPUT = "/home/pietro/Documenti/doorkey/src/prove/reward_subset_seed1_zero.csv"
OUTPUT = "/dev/stdout"

# direction vectors: 0=right, 1=down, 2=left, 3=up
DX = [1, 0, -1, 0]
DY = [0, 1, 0, -1]

def is_free(x, y):
    return 1 <= x <= 4 and 1 <= y <= 4

def reward(ax, ay, ad, gx, gy, gname, act):
    dist = abs(ax - gx) + abs(ay - gy)

    if act == "FORWARD":
        nx, ny = ax + DX[ad], ay + DY[ad]
        if not is_free(nx, ny):
            return -0.02
        nd = abs(nx - gx) + abs(ny - gy)
        d = dist - nd
        if gname == "door_open":
            return 0.15 if d > 0 else (-0.05 if d < 0 else 0.0)
        elif gname == "has_key":
            return 0.1 if d > 0 else (-0.05 if d < 0 else 0.0)
        else:
            return 0.1 if d > 0 else (-0.05 if d < 0 else 0.0)

    if act in ("LEFT", "RIGHT"):
        return -0.01

    if act == "PICKUP":
        if gname == "no_key":
            return -0.05
        return -0.02

    if act == "DROP":
        if gname == "no_key":
            return -0.05
        return -0.02

    if act == "TOGGLE":
        if gname == "has_key":
            return 0.3
        if gname == "door_open":
            return -0.02
        return -0.05

    if act == "DONE":
        return 0.0

    return -0.01

with open(INPUT) as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print("agent_x,agent_y,agent_dir,goal_pos,goal_name,action,reward")
for r in rows:
    ax, ay, ad = int(r["agent_x"]), int(r["agent_y"]), int(r["agent_dir"])
    gx, gy = ast.literal_eval(r["goal_pos"])
    gn, act = r["goal_name"], r["action"]
    rew = reward(ax, ay, ad, gx, gy, gn, act)
    print(f"{ax},{ay},{ad},\"({gx}, {gy})\",{gn},{act},{rew}")
