import functools
import dill
dill.settings['recurse'] = True


def f(a, b, c):  # without keywords
    pass


def g(a, b, c=2):  # with keywords
    pass


def h(a=1, b=2, c=3):  # without args
    pass


def test_functools():
    fp = functools.partial(f, 1, 2)
    gp = functools.partial(g, 1, c=2)
    hp = functools.partial(h, 1, c=2)
    bp = functools.partial(int, base=2)

    assert dill.pickles(fp, safe=True)
    assert dill.pickles(gp, safe=True)
    assert dill.pickles(hp, safe=True)
    assert dill.pickles(bp, safe=True)


if __name__ == '__main__':
    test_functools()