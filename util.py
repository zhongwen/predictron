"""Utils."""

class Colour():
  '''
  Borrowed from https://github.com/brendanator/predictron/blob/master/predictron/util.py
  '''
  NORMAL = '\033[0m'

  BLACK = '\033[30m'
  RED = '\033[31m'
  GREEN = '\033[32m'
  YELLOW = '\033[33m'
  BLUE = '\033[34m'
  MAGENTA = '\033[35m'
  CYAN = '\033[36m'
  WHITE = '\033[37m'

  @classmethod
  def highlight(cls, input_, success):
    if success:
      colour = Colour.GREEN
    else:
      colour = Colour.RED
    return colour + input_ + Colour.NORMAL