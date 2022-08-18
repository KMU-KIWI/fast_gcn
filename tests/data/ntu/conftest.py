def pytest_addoption(parser):
    parser.addoption("--ntu_path", type=str, default=".")
    parser.addoption("--debug", action="store_true")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.ntu_path
    if "ntu_path" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("ntu_path", [option_value])

    option_value = metafunc.config.option.debug
    if "debug" in metafunc.fixturenames:
        metafunc.parametrize("debug", [option_value])
