import retro
import numpy as np

print(retro.__path__)

# retro.make( Gamename, state )
# Gamename can be found in : retro/data/stable/game of choice
# State can be found in the previous folder
env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1.test')
env.reset()
buttons = env.buttons = ['B', 'LEFT', 'RIGHT']
print(buttons)
combos = env.button_combos
print(combos)

print("action space: ", env.action_space.n)  # this shows that Sonic is MultiBinary(12)
action_space_size = env.action_space.n
opt_policy = np.argmax([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print(opt_policy)
done = False
while not done:
    env.render()
    # action = buttons pressable for the game --> Genesis has 12 buttons so 12 actions possible
    action = [7]
    print(action)
    # opt_policy = np.argmax([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    env.step(action)

env.close()
