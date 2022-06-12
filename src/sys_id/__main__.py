"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """System Identification."""


if __name__ == "__main__":
    main(prog_name="sys-id")  # pragma: no cover
