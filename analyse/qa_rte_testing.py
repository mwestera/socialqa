from transformers import AutoTokenizer, AlbertForSequenceClassification,AutoModelForQuestionAnswering,AutoModelForSequenceClassification,AutoModel, AutoTokenizer 
import torch
import argparse
class QA:
    def __init__(self,tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
    def find_best_token_index(self, predictions, input_ids, tokenizer, skip_tokens):
        # Sort the predictions in descending order of probability and get the indices
        sorted_indices = torch.argsort(predictions, descending=True)
        for idx in sorted_indices[0]:  # Iterate over indices of sorted predictions
            # Ensure the idx is used to index input_ids correctly as a list
            token_ids = input_ids[idx].unsqueeze(0)  # Add a dimension to make it iterable
            token = tokenizer.convert_ids_to_tokens(token_ids)[0]
            if token not in skip_tokens:
                return idx.item(), predictions[0, idx].item()
        return None, None  # In case all tokens are skip tokens, which should not happen


    def __predict__(self,question,context):
        inputs = self.tokenizer(question, context, return_tensors='pt')
        outputs = self.model(**inputs)

        predictions_start = torch.softmax(outputs.start_logits, dim=1)
        predictions_end = torch.softmax(outputs.end_logits, dim=1)

        # Define skip tokens
        skip_tokens = ["[CLS]", "[SEP]"]

        # Find the best start and end indices, skipping over the special tokens
        answer_start, prob_start = self.find_best_token_index(predictions_start, inputs['input_ids'][0], self.tokenizer, skip_tokens)
        answer_end, prob_end = self.find_best_token_index(predictions_end, inputs['input_ids'][0], self.tokenizer, skip_tokens)
        # Ensure we have valid start and end
        if answer_start is not None and answer_end is not None and answer_start <= answer_end:
            # Convert tokens to answer string
            answer = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end + 1]))
            average_prob = (prob_start + prob_end) / 2
            print(answer, average_prob)
            return average_prob
        else:
            print("No valid answer found.")
class RTE:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def __predict__(self,sentence1, sentence2):
        inputs = self.tokenizer(sentence1, sentence2, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)

        # Process the outputs
        predictions = torch.softmax(outputs.logits, dim=1)
        labels = ['not_entailment', 'entailment']
        predicted_label = labels[torch.argmax(predictions).item()]
        return predictions[0,1]


def main():
    parser = argparse.ArgumentParser(description="Process input for QA and RTE models.")
    parser.add_argument('--prior', type=str,  required=True, help='Question for the QA model')
    parser.add_argument('--pivot', type=str, required=True, help='Context for the QA model... pivot')
    parser.add_argument('--posterior', type=str, required=True, help='posterior for the RTE model')

    # Parse arguments
    args = parser.parse_args()
    if(args.prior =="" or args.pivot =="" or args.posterior==""):
        print("No input provided")
        return 
    
    print("QA Model Inputs:")
    print(f"Question: {args.prior}")
    print(f"Pivot: {args.pivot}")
    
    print("\nRTE Model Inputs:")
    print(f"Pivot: {args.pivot}")
    print(f"Sentence1: {args.posterior}")

    # model albert
    model_name = "ahotrod/albert_xxlargev1_squad2_512" 
    model_qa = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer_qa = AutoTokenizer.from_pretrained(model_name)
    qa =QA(tokenizer_qa, model_qa)


    prob=qa.__predict__(args.prior,args.pivot)
    print("------------------------- QA OUTPUT --------------------------")
    print(f"Score QA: {prob}")

    # Model RTE
    tokenizer = AutoTokenizer.from_pretrained("ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")
    model = AutoModelForSequenceClassification.from_pretrained("ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli")
    rte = RTE(tokenizer, model)
    print("------------------------- RTE OUTPUT --------------------------")
    score=rte.__predict__(args.pivot,args.posterior)
    print(f"Score RTE: {score}")

if __name__=='__main__':
    main()
    '''
    Scores QA model default
    Results: 
    {
        'exact': 78.71010200723923, 
        'f1': 81.89228117126069, 
        'total': 6078, 
        'HasAns_exact': 75.39518900343643, 
        'HasAns_f1': 82.04167868004215, 
        'HasAns_total': 2910, 
        'NoAns_exact': 81.7550505050505, 
        'NoAns_f1': 81.7550505050505, 
        'NoAns_total': 3168, 
        'best_exact': 78.72655478775913, 
        'best_exact_thresh': 0.0, 
        'best_f1': 81.90873395178066, 
        'best_f1_thresh': 0.0
    }
'''

