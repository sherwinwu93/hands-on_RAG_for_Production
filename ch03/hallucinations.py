#pip install transformers
from transformers import pipeline, AutoTokenizer

example_pairs = [
    # Good summary
    {"article": "The woman is playing mario cart while resting on the couch",
     "summary": "The woman is playing a game resting"},

    # Bad summary: completely different summary
    {"article": "A person on a horse jumps over a broken down airplane.",
     "summary": "A person is at a diner, ordering an omelette."},

    # Bad Summary: 2kg vs 2lbs, one meter vs two meters
    {"article": "Goldfish are being caught weighing up to 2kg and koi carp up to 8kg and one metre in length",
     "summary": "Koi carp can be as heavy as 2lbs and as long as two meters"},

    # Bad Summary: article didn't mention estimated worth
    {
        "article": "The plants were found during the search of a warehouse near Ashbourne on Saturday morning. Police said they were in 'an elaborate grow house'. A man in his late 40s was arrested at the scene.",
        "summary": "Police have arrested a man in his late 40s after cannabis plants worth an estimated £100,000 were found in a warehouse near Ashbourne."}
]

prompt = "<pad> Determine if the hypothesis is true given the premise?\n\nPremise: {text1}\n\nHypothesis: {text2}"
input_pairs = [prompt.format(text1=pair['article'], text2=pair['summary']) for pair in example_pairs]
classifier = pipeline(
            "text-classification",
            model='vectara/hallucination_evaluation_model',
            tokenizer=AutoTokenizer.from_pretrained('google/flan-t5-base'),
            trust_remote_code=True
        )
full_scores = classifier(input_pairs, top_k=None) # List[List[Dict[str, float]]]
hhem_scores = [round(score_dict['score'],4) for score_for_both_labels in full_scores for score_dict in score_for_both_labels if score_dict['label'] == 'consistent']
print(hhem_scores)