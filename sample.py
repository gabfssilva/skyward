import skyward as sky


@sky.function
def get_ip() -> str:
    print("hihi")
    return "localhost"

@sky.main
def main() -> None:
    result = get_ip() >> sky
    print(result)
