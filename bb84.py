"""Standalone entry point for BB84 angle rules."""

from qkd_angles import protocol_main


def main(argv: list[str] | None = None) -> None:
    protocol_main("bb84", argv)


if __name__ == "__main__":
    main()
