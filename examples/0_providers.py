import json
from dataclasses import asdict

import skyward as sky

def instances(provider: sky.Provider):
    return json.dumps(list(map(lambda spec: asdict(spec), provider.available_instances())), indent=2)

if __name__ == '__main__':
    print(instances(sky.AWS()))
    print(instances(sky.Verda()))
