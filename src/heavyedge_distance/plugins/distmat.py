"""Commands to compute distance matrix between profile functions."""

import pathlib

from heavyedge.cli.command import Command, register_command

PLUGIN_ORDER = 1.0


@register_command("dist-frechet", "Fréchet distance matrix")
class FretchetDistCommand(Command):
    def add_parser(self, main_parser):
        dist = main_parser.add_parser(
            self.name,
            description="Compute 1-D Fréchet distance matrix of profile functions.",
            epilog="To compute distance matrix of Y1 to itself, do not pass Y2.",
        )
        dist.add_argument(
            "Y1",
            type=pathlib.Path,
            help="Path to the first profiles in 'ProfileData' structure.",
        )
        dist.add_argument(
            "Y2",
            type=pathlib.Path,
            nargs="?",
            help=(
                "Path to the second profiles in 'ProfileData' structure. "
                "Set to Y1 if not passed."
            ),
        )
        dist.add_argument(
            "--batch-size",
            type=int,
            help="Batch size to load data. If not provided, loads entire profiles.",
        )
        dist.add_argument(
            "--n-jobs",
            type=int,
            help=(
                "Number of parallel workers. "
                "If not passed, tries HEAVYEDGE_MAX_WORKERS environment variable."
            ),
        )
        dist.add_argument(
            "-o", "--output", type=pathlib.Path, help="Output npy file path"
        )

    def run(self, args):
        import numpy as np
        from heavyedge import ProfileData

        from heavyedge_distance.api import distmat_frechet

        file1 = ProfileData(args.Y1)
        if args.Y2 is not None:
            file2 = ProfileData(args.Y2)
        else:
            file2 = None
        out = args.output.expanduser()

        self.logger.info(f"Computing Fréchet distance matrix: {out}")

        def logger(msg):
            self.logger.info(f"{out} : {msg}")

        D = distmat_frechet(file1, file2, args.batch_size, args.n_jobs, logger)
        np.save(out, D)
        self.logger.info(f"Saved {out}.")
