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

    def testEvaluation_4(self):
        golden_list = [['B-TAR', 'I-TAR', 'O', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0.0)

    def testEvaluation_5(self):
        golden_list = [
            ['B-TAR', 'O', 'O', 'B-HYP', 'I-HYP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TAR', 'O', 'O', 'O', 'O', 'B-HYP', 'I-HYP', 'I-HYP', 'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O']]
        predict_list = [
            ['B-TAR', 'O', 'O', 'O', 'B-HYP', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'], [
                'O', 'I-TAR', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TAR', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                'O',
                'O', 'O', 'O', 'I-HYP', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(2 / 4, 2 / 3))

    def testEvaluation_6(self):
        golden_list = [['B-TAR', 'I-TAR', 'B-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'I-TAR', 'I-TAR', 'O'], ['I-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(1 / 5, 1 / 1))

    def testEvaluation_7(self):
        golden_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'B-TAR', 'I-TAR', 'O'], ['I-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)


    def testEvaluation_8(self):
        golden_list = [['B-TAR', 'B-TAR', 'I-TAR', 'B-HYP', 'O', 'O'], ['B-TAR', 'O', 'O', 'B-HYP', '0', 'I-HYP']]
        predict_list = [['B-TAR', 'B-TAR', 'I-TAR', 'O', 'B-TAR', 'I-TAR'], ['I-TAR', 'O', 'B-HYP', 'B-HYP', 'B-TAR']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(3 / 6, 3 / 5))


    def testEvaluation_9(self):
        golden_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O']]
        predict_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 1)

    def testEvaluation_10(self):
        golden_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'I-TAR', 'B-TAR']]
        predict_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'B-TAR', 'I-TAR']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)

    def testEvaluation_11(self):
        golden_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', '0', '0']]
        predict_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'B-TAR', 'I-TAR']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)


    def testEvaluation_12(self):
        golden_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'B-TAR', 'I-TAR', 'O'], ['I-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)

    def testEvaluation_13(self):
        golden_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'B-TAR', 'I-TAR', 'B-HYP'], ['I-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(1/3,1/4))



if __name__ == '__main__':
    unittest.main()
