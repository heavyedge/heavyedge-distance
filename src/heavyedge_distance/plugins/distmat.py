"""Commands to compute distance matrix between profile functions."""

import pathlib

from heavyedge.cli.command import Command, register_command

PLUGIN_ORDER = 1.0


@register_command("dist-euclidean", "Euclidean distance matrix")
class EuclideanDistCommand(Command):
    def add_parser(self, main_parser):
        dist = main_parser.add_parser(
            self.name,
            description="Compute 1-D Euclidean distance matrix of profile functions.",
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
            "-o", "--output", type=pathlib.Path, help="Output npy file path"
        )

    def run(self, args):
        import numpy as np
        from heavyedge import ProfileData

        from heavyedge_distance.api import distmat_euclidean

        file1 = ProfileData(args.Y1)
        if args.Y2 is not None:
            file2 = ProfileData(args.Y2)
        else:
            file2 = None
        out = args.output.expanduser()

        self.logger.info(f"Writing {out}")

        D = distmat_euclidean(
            file1,
            file2,
            args.batch_size,
            lambda msg: self.logger.info(f"{out} : {msg}"),
        )
        np.save(out, D)

        file1.close()
        if file2 is not None:
            file2.close()
        self.logger.info(f"Saved {out}.")


@register_command("dist-wasserstein", "Wasserstein distance matrix")
class WassersteinDistCommand(Command):
    def add_parser(self, main_parser):
        dist = main_parser.add_parser(
            self.name,
            description="Compute Wasserstein distance matrix of area-scaled profiles.",
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
        dist.add_config_argument(
            "--grid-num",
            type=int,
            help="Number of grids to sample quantile functions.",
        )
        dist.add_argument(
            "--batch-size",
            type=int,
            help="Batch size to load data. If not provided, loads entire profiles.",
        )
        dist.add_argument(
            "-o", "--output", type=pathlib.Path, help="Output npy file path"
        )

    def run(self, args):
        import numpy as np
        from heavyedge import ProfileData

        from heavyedge_distance.api import distmat_wasserstein

        file1 = ProfileData(args.Y1)
        if args.Y2 is not None:
            file2 = ProfileData(args.Y2)
        else:
            file2 = None
        out = args.output.expanduser()

        self.logger.info(f"Writing {out}")

        t = np.linspace(0, 1, args.grid_num)
        D = distmat_wasserstein(
            t,
            file1,
            file2,
            args.batch_size,
            lambda msg: self.logger.info(f"{out} : {msg}"),
        )
        np.save(out, D)

        file1.close()
        if file2 is not None:
            file2.close()
        self.logger.info(f"Saved {out}.")


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

        self.logger.info(f"Writing {out}")

        self.logger.info(f"Computing Fréchet distance matrix: {out}")

        D = distmat_frechet(
            file1,
            file2,
            args.batch_size,
            args.n_jobs,
            lambda msg: self.logger.info(f"{out} : {msg}"),
        )
        np.save(out, D)

        file1.close()
        if file2 is not None:
            file2.close()
        self.logger.info(f"Saved {out}.")
