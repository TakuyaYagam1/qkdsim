"""Standalone entry point for B92 angle rules."""

from qkd_angles import protocol_main


def main(argv: list[str] | None = None) -> None:
    protocol_main("b92", argv)


if __name__ == "__main__":
    main()
