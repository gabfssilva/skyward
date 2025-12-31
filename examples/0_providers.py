import json
from dataclasses import asdict

from skyward import AWS, Verda, Provider

def instances(provider: Provider):
    return json.dumps(list(map(lambda spec: asdict(spec), provider.available_instances())), indent=2)

if __name__ == '__main__':
    print(instances(AWS()))
    print(instances(Verda()))
