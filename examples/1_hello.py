from skyward import *

@compute
def remote_sum(x: int, y: int) -> int:
    print("That's one expensive sum.")
    return x + y

if __name__ == '__main__':
    with ComputePool(provider=Verda(), cpu=2, memory="4GB") as pool:
        result = remote_sum(x=1, y=2) >> pool
        print(result)
