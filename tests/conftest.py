def pytest_benchmark_scale_unit(config, unit, benchmarks, best, worst, sort):
    if unit == 'seconds':
        prefix = 'millisec'
        scale = 1000
    elif unit == 'operations':
        prefix = ''
        scale = 1
    else:
        raise RuntimeError("Unexpected measurement unit %r" % unit)
    return prefix, scale
