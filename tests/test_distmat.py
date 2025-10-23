import subprocess


def test_distmat_frechet(tmp_path, tmp_profile_type2_path, tmp_profile_type3_path):
    subprocess.run(
        [
            "heavyedge",
            "dist-frechet",
            tmp_profile_type2_path,
            "-o",
            tmp_path / "fdist.npy",
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
            tmp_path / "fdist.npy",
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
            tmp_path / "fdist.npy",
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
            tmp_path / "fdist.npy",
        ],
        capture_output=True,
        check=True,
    )
