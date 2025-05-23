# ~/mved_probabilistic_surgery/scripts/utils/eval_utils.py
def calculate_accuracy(predictions: list, ground_truths: list) -> float:
    if len(predictions) != len(ground_truths):
        # Consider if this should raise error or return specific value
        print(f"Warning: Predictions ({len(predictions)}) and ground truths ({len(ground_truths)}) have different lengths.")
        return 0.0 # Or handle as appropriate
    if not predictions: # Handle empty case
        return 0.0

    correct = 0
    for p, gt in zip(predictions, ground_truths):
        # Assuming predictions are 0 or 1, and ground_truths are 1 (for simple S-P-O task)
        # Or, if predictions are actual generated strings and GTs are expected strings:
        # if str(p).strip().lower() == str(gt).strip().lower():
        #    correct += 1
        if p == gt: # Generic comparison
             correct +=1
    return correct / len(predictions)