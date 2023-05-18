#strat from here

import sys
from multiprocessing import freeze_support

from setproctitle import setproctitle

from learning.train import train

if __name__ == "__main__":
    freeze_support()
    setproctitle("learning")

    train("trading", "trading")
