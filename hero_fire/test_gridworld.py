from GridWorld import gameEnv
import matplotlib.pyplot as plt

env = gameEnv(size=5)
a = env.renderEnv()
env.moveChar(0)
plt.show()
