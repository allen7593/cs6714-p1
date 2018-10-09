import unittest

from todo import evaluate, f1_score


class EvaluationTest(unittest.TestCase):
    def testEvaluation_1(self):
        golden_list = [['B-TAR', 'I-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'O', 'O', 'O'], ['B-TAR', 'O', 'B-HYP', 'I-HYP']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(1 / 4, 1 / 3))

    def testEvaluation_2(self):
        golden_list = [['B-TAR', 'I-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['O', 'O', 'O', 'O'], ['B-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0.4)

    def testEvaluation_3(self):
        golden_list = [['B-TAR', 'I-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['O', 'O', 'I-TAR', 'O'], ['B-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0.4)
