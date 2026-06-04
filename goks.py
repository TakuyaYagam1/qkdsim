"""Standalone entry point for GOKS angle rules."""

from qkd_angles import protocol_main


def main(argv: list[str] | None = None) -> None:
    protocol_main("goks", argv)


if __name__ == "__main__":
    main()
