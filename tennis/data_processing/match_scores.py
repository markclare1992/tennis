"""Functions to parse and process match scores."""
from typing import Any, Optional, Union
import pandas as pd


def parse_set_score(set_score: str) -> dict[str, Union[int, str, None]]:
    """
    Parse a single set score and extract points for player 1 and player 2,
    as well as tiebreaker points if present.

    Parameters:
    - set_score (str): The set score to parse.

    Returns:
    - Dict[str, Union[int, str, None]]: A dictionary containing the parsed set score details.
    """
    set_score = set_score.strip()
    if set_score in {'RET', 'DEF', 'W/O'}:
        return {
            'p1': None,
            'p2': None,
            'winner': None,
            'tiebreaker': None,
            'status': set_score
        }
    tiebreaker = None
    if '(' in set_score:
        set_score, tiebreaker = set_score.split('(')
        set_score = set_score.strip()
        tiebreaker = int(tiebreaker.replace(')', '').strip())
    p1, p2 = map(int, set_score.split('-'))

    # Determine if the set is complete
    if (p1 >= 6 and (p1 - p2) >= 2) or p1 >= 7 or (
            p2 >= 6 and (p2 - p1) >= 2) or p2 >= 7:
        status = 'complete'
        winner = 'p1' if p1 > p2 else 'p2'
    else:
        status = 'incomplete'
        winner = None

    return {
        'p1': p1,
        'p2': p2,
        'winner': winner,
        'tiebreaker': tiebreaker,
        'status': status
    }


def handle_special_marker(parsed_sets: dict[str, Any], i: int, marker: str) -> None:
    """
    Handle the special markers (RET, DEF, W/O) by updating the parsed sets accordingly.

    Parameters:
    - parsed_sets (Dict[str, Any]): The parsed sets dictionary.
    - i (int): The current set index.
    - marker (str): The special marker (RET, DEF, W/O).
    """
    if i > 1 and parsed_sets[f'set_{i - 1}_status'] == 'incomplete':
        parsed_sets[f'set_{i - 1}_status'] = marker
    else:
        parsed_sets[f'set_{i}_status'] = marker
        parsed_sets[f'set_{i}_p1'] = None
        parsed_sets[f'set_{i}_p2'] = None
        parsed_sets[f'set_{i}_tiebreaker'] = None
        parsed_sets[f'set_{i}_winner'] = None


def ensure_all_sets(parsed_sets: dict[str, Any], match_ended: bool) -> None:
    """
    Ensure that all five sets are represented and handle cases where the match ends early.

    Parameters:
    - parsed_sets (Dict[str, Any]): The parsed sets dictionary.
    - match_ended (bool): Flag indicating if the match ended early.
    """
    for i in range(1, 6):
        parsed_sets.setdefault(f'set_{i}_p1', None)
        parsed_sets.setdefault(f'set_{i}_p2', None)
        parsed_sets.setdefault(f'set_{i}_winner', None)
        parsed_sets.setdefault(f'set_{i}_tiebreaker', None)
        if match_ended and f'set_{i}_status' not in parsed_sets:
            parsed_sets[f'set_{i}_status'] = 'NA'
        else:
            parsed_sets.setdefault(f'set_{i}_status', 'incomplete')


def parse_scores(score: str) -> Optional[dict[str, Any]]:
    """
    Parse the match score string into a structured format.

    Parameters:
    - score (str): The score string to parse.

    Returns:
    - Optional[Dict[str, Any]]: A dictionary containing the parsed match details,
      or None if the parsing failed.
    """
    sets = [set_score.strip() for set_score in score.split(' ') if set_score.strip()]
    parsed_sets: dict[str, Any] = {}
    match_ended = False
    p1_set_wins = 0
    p2_set_wins = 0

    try:
        for i, set_score in enumerate(sets, 1):
            if set_score in {'RET', 'DEF', 'W/O'}:
                handle_special_marker(parsed_sets, i, set_score)
                match_ended = True
                break

            set_result = parse_set_score(set_score)
            parsed_sets[f'set_{i}_p1'] = set_result['p1']
            parsed_sets[f'set_{i}_p2'] = set_result['p2']
            parsed_sets[f'set_{i}_tiebreaker'] = set_result['tiebreaker']
            parsed_sets[f'set_{i}_status'] = set_result['status']

            if set_result['status'] == 'complete':
                parsed_sets[f'set_{i}_winner'] = set_result['winner']
                if set_result['winner'] == 'p1':
                    p1_set_wins += 1
                elif set_result['winner'] == 'p2':
                    p2_set_wins += 1
            else:
                parsed_sets[f'set_{i}_winner'] = None

        # Ensure all five sets are represented and handle cases where match ends early
        ensure_all_sets(parsed_sets, match_ended)

        # Determine if the match was completed
        parsed_sets['match_completed'] = not match_ended

        # Determine the overall winner if the match was completed
        if not match_ended:
            if p1_set_wins > p2_set_wins:
                parsed_sets['match_winner'] = 'p1'
            elif p2_set_wins > p1_set_wins:
                parsed_sets['match_winner'] = 'p2'
            else:
                parsed_sets['match_winner'] = None
        else:
            parsed_sets['match_winner'] = None

    except Exception as e:
        print(f"Error parsing score '{score}': {e}")
        return None

    return parsed_sets


def process_match_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the match scores in the DataFrame and add structured match details.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing match scores.

    Returns:
    - pd.DataFrame: The DataFrame with structured match details added.
    """
    parsed_scores = df['score'].apply(parse_scores).dropna()
    parsed_scores_df = pd.DataFrame(parsed_scores.tolist())
    df = df.loc[parsed_scores.index].join(parsed_scores_df)
    df.drop(columns=['score'], inplace=True)
    return df