peremennaya = 1

def doSomething1(input):
    print(input)
    x = 1
    y = 2
    z = x + y
    z = helper(z)
    return z

def helper(x):
    return x + 1