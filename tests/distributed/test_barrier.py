def test_barrier_single_wait(registry):
    proxy = registry.barrier("test_barrier_single", n=1)
    proxy.wait()


def test_barrier_reset(registry):
    proxy = registry.barrier("test_barrier_reset", n=1)

    proxy.wait()
    proxy.reset()
    proxy.wait()
