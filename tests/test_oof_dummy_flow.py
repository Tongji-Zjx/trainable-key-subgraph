from __future__ import absolute_import, print_function

import unittest

from scripts.check_oof_dummy_flow import main


class OofDummyFlowTest(unittest.TestCase):
    def test_complete_dummy_flow(self):
        self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
