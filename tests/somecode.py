# Sample Python code used to create some tests.

class SomeCode:
    """A class."""

    def __init__(self):
        pass

    def foo(self, n):
        n += 1

        for i in range(10):
            n += i

        n += 1
        n += 1

        return n

    @staticmethod
    def bar(x):
        """a decorated function."""
        return sum([n for n in range(x)])

def globalDef():
    """A globally defined function."""
    return 42

def globalDef2():
    """Something else, globally defined."""

    def inner():
        """An inner function that really can't be tested separately."""
        return 111+222+333

    return inner()

if __name__ == "__main__":
    """Something that always executes."""
    assert globalDef() == 42
