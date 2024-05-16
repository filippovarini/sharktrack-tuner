import unittest
from pathlib import Path
from path_resolver import construct_new_folder

class TestComputeSpeciesMaxN(unittest.TestCase):
    def test__path_does_not_exist(self):
        # Mock the exists method to always return False
        path = Path('/some/non/existent/folder')
        result = construct_new_folder(path)
        self.assertEqual(result, path)

    def test__path_exists(self):
        path = Path("utils")
        result = construct_new_folder(path)
        expected = Path("utils1")
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()