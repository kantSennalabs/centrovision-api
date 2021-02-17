import sys, time


def print_out(s, arg=None):
    sys.stdout.write(f'{s} {arg if arg != None else ""}')
    sys.stdout.flush()
    time.sleep(0.05)
