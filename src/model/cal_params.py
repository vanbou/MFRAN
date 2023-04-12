from torchstat import stat
from option import args
from mfran import MFRAN
model = MFRAN(args)


stat(model, input_size=(3, 48, 48))


