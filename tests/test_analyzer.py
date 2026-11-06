import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from source.analyzer.modes import check_chess_file, check_fen
from source.analyzer.parser import parse_file, parse_line
from source.analyzer.save import save_as_csv, save_as_img
from source.common.constants import LABELS


VALID_FEN = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = REPOSITORY_ROOT / "my_torch_analyzer"


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        check=False,
    )


class AnalyzerParserTests(unittest.TestCase):
    def test_parser_accepts_and_normalizes_all_valid_labels(self):
        for label in LABELS.values():
            with self.subTest(label=label):
                entry = parse_line(f"{VALID_FEN} {label.upper()}")
                self.assertEqual(entry.label, label)

    def test_parser_rejects_malformed_fen_fields(self):
        invalid_fens = (
            "4k3/8/8/8/8/8/8/4K3 w - - 0",
            "4k3/8/8/8/8/8/8/4K3 x - - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w KK - 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - z9 0 1",
            "4k3/8/8/8/8/8/8/4K3 w - - invalid 1",
            "4k3/44/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/4K20 w - - 0 1",
        )
        for fen in invalid_fens:
            with self.subTest(fen=fen), self.assertRaises(ValueError):
                parse_line(fen)

    def test_parser_accepts_valid_empty_square_counts(self):
        entry = parse_line(VALID_FEN)
        self.assertEqual(entry.placement, "4k3/8/8/8/8/8/8/4K3")

    def test_parser_rejects_unknown_labels(self):
        with self.assertRaisesRegex(ValueError, "invalid label"):
            parse_line(f"{VALID_FEN} Unknown")

    def test_file_validation_and_parser_accept_the_same_labeled_line(self):
        with tempfile.TemporaryDirectory() as directory:
            chess_file = Path(directory) / "boards.txt"
            chess_file.write_text(
                f"{VALID_FEN} Check White\n",
                encoding="utf-8",
            )

            check_chess_file(str(chess_file), need_label=True)
            self.assertEqual(parse_file(str(chess_file))[0].label, "Check White")

    def test_file_validation_reports_parser_errors_with_line_number(self):
        with tempfile.TemporaryDirectory() as directory:
            chess_file = Path(directory) / "boards.txt"
            chess_file.write_text(
                f"{VALID_FEN} Nothing\n{VALID_FEN} Unknown\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as exc:
                check_chess_file(str(chess_file), need_label=True)
            self.assertIn(f"{chess_file}:2:", str(exc.exception))

    def test_file_validation_requires_labels_only_for_training(self):
        with tempfile.TemporaryDirectory() as directory:
            chess_file = Path(directory) / "boards.txt"
            chess_file.write_text(f"{VALID_FEN}\n", encoding="utf-8")

            check_chess_file(str(chess_file), need_label=False)
            with self.assertRaisesRegex(ValueError, "expected output is missing"):
                check_chess_file(str(chess_file), need_label=True)

    def test_check_fen_uses_the_same_fen_validation(self):
        with self.assertRaisesRegex(ValueError, "each side needs exactly one king"):
            check_fen("4k3/8/8/8/8/8/8/8 w - - 0 1".split())


class AnalyzerArtifactTests(unittest.TestCase):
    def test_training_artifacts_are_written_to_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "artifacts"

            csv_path = save_as_csv([1.0, 0.5], output_dir)
            image_path = save_as_img([1.0, 0.5], output_dir)

            self.assertTrue(csv_path.is_file())
            self.assertTrue(image_path.is_file())
            self.assertEqual(
                csv_path.read_text(encoding="utf-8").splitlines(),
                ["epoch,loss", "1,1.0", "2,0.5"],
            )

class AnalyzerCliTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.loadfile = self.directory / "network.dat"
        self.trainfile = self.directory / "train.txt"
        self.predictfile = self.directory / "predict.txt"
        self.savefile = self.directory / "saved-network.dat"
        self.loadfile.write_text("network placeholder\n", encoding="utf-8")
        self.trainfile.write_text(f"{VALID_FEN} Nothing\n", encoding="utf-8")
        self.predictfile.write_text(f"{VALID_FEN}\n", encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_help_argument(self):
        result = run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("USAGE", result.stdout)
        self.assertEqual(result.stderr, "")

    def test_help_rejects_extra_arguments(self):
        result = run_cli("--help", "extra")
        self.assertEqual(result.returncode, 84)
        self.assertIn("unknown argument", result.stderr.lower())

    def test_train_arguments_accept_valid_inputs(self):
        result = run_cli("--train", str(self.loadfile), str(self.trainfile))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_train_arguments_report_invalid_chess_file_with_line_number(self):
        bad_trainfile = self.directory / "bad-train.txt"
        bad_trainfile.write_text(
            f"{VALID_FEN} Nothing\n{VALID_FEN} Unknown\n",
            encoding="utf-8",
        )
        result = run_cli("--train", str(self.loadfile), str(bad_trainfile))
        self.assertEqual(result.returncode, 84)
        self.assertIn(f"{bad_trainfile}:2:", result.stderr)

    def test_predict_arguments_accept_valid_inputs(self):
        result = run_cli("--predict", str(self.loadfile), str(self.predictfile))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_predict_arguments_report_missing_files(self):
        result = run_cli(
            "--predict",
            str(self.loadfile),
            str(self.directory / "missing.txt"),
        )
        self.assertEqual(result.returncode, 84)
        self.assertIn("does not actually exist", result.stderr)

    def test_save_arguments_accept_valid_inputs(self):
        result = run_cli(
            "--train",
            "--save",
            str(self.savefile),
            str(self.loadfile),
            str(self.trainfile),
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_save_arguments_reject_missing_directory(self):
        result = run_cli(
            "--train",
            "--save",
            str(self.directory / "missing" / "network.dat"),
            str(self.loadfile),
            str(self.trainfile),
        )
        self.assertEqual(result.returncode, 84)
        self.assertIn("invalid SAVEFILE", result.stderr)

    def test_save_arguments_reject_misplaced_save_flag(self):
        result = run_cli(
            "--train",
            str(self.loadfile),
            "--save",
            str(self.savefile),
            str(self.trainfile),
        )
        self.assertEqual(result.returncode, 84)
        self.assertIn("Expected '--save SAVEFILE'", result.stderr)


if __name__ == "__main__":
    unittest.main()
