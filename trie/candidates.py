import pandas as pd

def score_candidates(candidates: dict) -> dict:
    df = pd.read_csv('../data/vocab_final.csv')

    final_scores = {}
    for candidate in candidates:
        final_scores[candidate] = spatial_score(candidate, candidates) * language_score(candidate, candidates, df)
    
    return final_scores
    


def spatial_score(candidate: str, candidates: dict) -> float:
    idx = 10 if len(candidates) > 10 else len(candidates)
    return candidates[candidate][0] / sum(sorted([candidates[c][0] for c in candidates], reverse=True)[:idx])

def language_score(candidate: str, candidates: dict, df: pd.DataFrame) -> float:
    idx = 10 if len(candidates) > 10 else len(candidates)
    # get the top 10 'log_count' values
    top_values = sorted(df['log_count'], reverse=True)[:idx]
    # get the log_count value for the candidate
    candidate_value = df[df['word'] == candidate]['log_count']
    candidate_value = candidate_value.values[0] if len(candidate_value) > 0 else 0

    return candidate_value / sum(top_values)