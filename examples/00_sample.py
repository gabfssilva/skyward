import asyncio
import dataclasses
import json

import skyward as sky

async def main():
    p = await sky.AWS().create_provider()
    offers = await sky.offers([p])
    all_offers = await offers.cpu_only().vcpus(8).cheapest(5)
    print(json.dumps(list(map(lambda v: dataclasses.asdict(v), all_offers)), indent=2))

if __name__ == '__main__':
    asyncio.run(main())