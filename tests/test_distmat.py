import subprocess


def test_distmat_euclidean(tmp_path, tmp_profile_type2_path, tmp_profile_type3_path):
    subprocess.run(
        [
            "heavyedge",
            "dist-euclidean",
            tmp_profile_type2_path,
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        [
            "heavyedge",
            "dist-euclidean",
            tmp_profile_type2_path,
            "--batch-size",
            "1",
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        [
            "heavyedge",
            "dist-euclidean",
            tmp_profile_type2_path,
            tmp_profile_type3_path,
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        [
            "heavyedge",
            "dist-euclidean",
            tmp_profile_type2_path,
            tmp_profile_type3_path,
            "--batch-size",
            "1",
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )


def test_distmat_frechet(tmp_path, tmp_profile_type2_path, tmp_profile_type3_path):
    subprocess.run(
        [
            "heavyedge",
            "dist-frechet",
            tmp_profile_type2_path,
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        [
            "heavyedge",
            "dist-frechet",
            tmp_profile_type2_path,
            "--batch-size",
            "1",
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        [
            "heavyedge",
            "dist-frechet",
            tmp_profile_type2_path,
            tmp_profile_type3_path,
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )

    subprocess.run(
        [
            "heavyedge",
            "dist-frechet",
            tmp_profile_type2_path,
            tmp_profile_type3_path,
            "--batch-size",
            "1",
            "-o",
            tmp_path / "dist.npy",
        ],
        capture_output=True,
        check=True,
    )
