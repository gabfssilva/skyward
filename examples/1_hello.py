from skyward import *

@compute
def remote_sum(x: int, y: int) -> int:
    print("hello from AWS!")
    print("let's wait for something, just for science")
    return x + y

if __name__ == '__main__':
    with ComputePool(provider=AWS(), spot='always') as pool:
        result = remote_sum(x=1, y=2) >> pool
        print(result)
