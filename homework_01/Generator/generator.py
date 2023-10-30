def gener(a):
    while True:
        a = a + "Meow "
        yield a

b = ""
b = gener(b)
print(next(b))