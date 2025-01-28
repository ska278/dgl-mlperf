"""
This package contains DistGNN and Libra based graph partitioning tools.
"""

mode = 'orig'
def set_mode(mode_):
    global mode
    mode = mode_

def get_mode():
    return mode

from . hels import DGLBlockPush
