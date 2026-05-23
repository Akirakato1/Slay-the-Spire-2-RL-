from scripts.live_smoke import build_parser


def test_live_smoke_defaults_to_read_only():
    args = build_parser().parse_args([])

    assert args.base_url == "http://localhost:15526"
    assert args.execute_end_turn is False
