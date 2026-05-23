def test_live_package_imports():
    import sts2_rl.live as live

    assert isinstance(live.__all__, list)
