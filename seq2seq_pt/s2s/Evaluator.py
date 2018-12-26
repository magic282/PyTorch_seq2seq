from PyRouge.Rouge import Rouge
import s2s


class Evaluator(object):
    def __init__(self):
        pass

    def evaluate(self):
        raise NotImplemented


class RougeEvaluator(Evaluator):
    def __init__(self, src_file: str, ref_file: str, translator: s2s.Translator):
        super(RougeEvaluator).__init__()
        self.src_file: str = src_file
        self.ref_file: str = ref_file
        self.rouge_calculator = Rouge.Rouge()
        self.translator = translator

    def evaluate(self):
        system_outputs = self.translator.translate_small_file(self.src_file, self.ref_file)
        refs = []
        with open(self.ref_file, 'w', encoding='utf-8') as reader:
            for line in reader:
                refs.append(line.strip())
        normed_output = [x.strip() for x in system_outputs]
        scores = self.rouge_calculator.compute_rouge(refs, normed_output)
        tune_score = scores['rouge-2']['f']
        return tune_score, scores, system_outputs, normed_output


class CNNDMRougeEvaluator(RougeEvaluator):
    def __init__(self, src_file: str, ref_file: str, translator: s2s.Translator):
        super().__init__(src_file, ref_file, translator)

    def evaluate(self):
        system_outputs = self.translator.translate_small_file(self.src_file, self.ref_file)
        refs = []
        with open(self.ref_file, 'r', encoding='utf-8') as reader:
            for line in reader:
                refs.append(line.strip().replace('<t>', '').replace('</t>', '').replace('[[', '').replace(']]', ''))
        normed_output = [x.replace('<t>', '').replace('</t>', '').replace('[[', '').replace(']]', '') for x in
                         system_outputs]
        scores = self.rouge_calculator.compute_rouge(refs, normed_output)
        tune_score = scores['rouge-2']['f'][0]
        return tune_score, scores, system_outputs, normed_output


class BleuEvaluator(Evaluator):
    def __init__(self):
        super(BleuEvaluator).__init__()

    def evaluate(self):
        raise NotImplemented
