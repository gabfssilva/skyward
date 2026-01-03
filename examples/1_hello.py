import skyward as sky

@sky.compute
def remote_sum(x: int, y: int) -> int:
    print("That's one expensive sum.")
    return x + y

if __name__ == '__main__':
    with sky.ComputePool(provider=sky.AWS(), machine='t4g.small') as pool:
        result = remote_sum(x=1, y=2) >> pool
        print(result)
