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
             'O'], 
            ['O', 'I-TAR', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TAR', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
                'O',
                'O', 'O', 'O', 'I-HYP', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(2 / 3, 2 / 4))

    def testEvaluation_6(self):
        golden_list = [['B-TAR', 'I-TAR', 'B-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'I-TAR', 'I-TAR', 'O'], ['I-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(0/ 5, 0 / 1))

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


    # both lists don't have any results
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


    # random test case
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

    # no any relevent
    # retrived some 
    def testEvaluation_11(self):
        golden_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'O', 'O']]
        predict_list = [
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
             'O', 'O', 'O', 'O', 'O', 'B-TAR', 'I-TAR']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)

    # sentence of different length
    def testEvaluation_12(self):
        golden_list  = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-HYP','I-HYP','O','B-TAR','O'],['O', 'O', 'B-HYP']]
        predict_list = [['O', 'O', 'O', 'O'], ['B-HYP','I-HYP','O','O','B-TAR'],['O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(1/2,1/5))

    # no any retrived 
    def testEvaluation_13(self):
        golden_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['O', 'O', 'O', 'O'], ['O', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)


    # invalid lists
    def testEvaluation_14(self):
        golden_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['O'],['B-TAR', 'I-TAR', 'O', 'B-HYP']]
        predict_list = [['I-TAR', 'B-TAR', 'I-TAR', 'B-HYP'], ['B-TAR'],['I-TAR', 'O', 'O', 'O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(1/3,1/4))

    # perfect match
    def testEvaluation_15(self):
        golden_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(4/4,4/4))

    def testEvaluation_16(self):
        golden_list = [['B-TAR', 'I-TAR', 'I-TAR', 'B-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]
        predict_list = [['B-TAR', 'I-TAR', 'I-TAR', 'I-HYP'], ['B-TAR', 'O', 'O', 'B-HYP']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, f1_score(3/3,3/4))

    def testEvaluation_17(self):
        golden_list = [['B-TAR']]
        predict_list = [['O']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)

    def testEvaluation_17(self):
        golden_list = [['O']]
        predict_list = [['B-TAR']]

        f1 = evaluate(golden_list, predict_list)
        self.assertEqual(f1, 0)






if __name__ == '__main__':
    unittest.main()
