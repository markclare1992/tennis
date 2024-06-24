"""
Functionality for markov transition matrix model.

Uses the probability of each player winning a point on their respective serves,
transition matrices and final transition matrices to calculate the probability of
winning a set and match.

Notes:
    The code and model can be drastically improved. Basic assumptions are made i.e,
    for tiebreaks and for final set format. The code is not optimized for performance.
"""

from dataclasses import dataclass
from typing import Generator

import numpy as np

DEFAULT_FIRST_SERVER = 1


@dataclass
class TennisParameters:
    """
    Dataclass to hold information on player 1 serve winning probability and
     player 2 serve winning probability."""

    player_one_point_on_serve_prob: float
    player_two_point_on_serve_prob: float


def service_game_winning_prob(service_win_prob: float) -> float:
    """
    Calculate the probability of winning a service game given the probability of
    winning a point on serve.

    Parameters:
        - service_win_prob (float): Probability of winning a point on serve.

    Returns:
        - float: Probability of winning a service game.
    """
    return (
        service_win_prob**4
        + 4 * (1 - service_win_prob) * (service_win_prob**4)
        + 10 * (1 - service_win_prob) ** 2 * (service_win_prob**4)
        + (20 * (1 - service_win_prob) ** 3 * (service_win_prob**5))
        / (1 - 2 * (1 - service_win_prob) * (service_win_prob))
    )


def state_to_index_set(a: int, b: int) -> int:
    """Convert the state (a, b) to an index in the transition matrix.
    Possible states are 0-0 ->6-6, and 7-5,5-7,7-6,6-7
    """

    return a + 7 * b + (b > 5) - 4 * (b > 6)


def index_to_state_set(index: int) -> tuple[int, int]:
    """Convert the index in the transition matrix to a state (a, b)."""
    all_states = [(a, b) for a in range(8) for b in range(8) if not (a == 7 and b < 5)]
    all_states_to_index = {state_to_index_set(a, b): (a, b) for a, b in all_states}
    return all_states_to_index[index]


def is_winning_set_score(a: int, b: int) -> bool:
    """Check if the set score (a, b) is a winning set score."""
    if b > a:
        return is_winning_set_score(b, a)
    if a == 6 and b < 5:
        return True
    if a == 7 and b in [5, 6]:
        return True
    return False


def is_p1_serving(a, b, first_server):
    """Check if player 1 is serving in the current game."""
    if first_server == 1:
        return (a + b) % 2 == 0
    return (a + b) % 2 == 1


def tie_break_winning_prob(p1_serve_win_prob: float, p2_serve_win_prob: float) -> float:
    """
    Estimate for winning tiebreak probability, implementation should be improved.
    Should follow implementation of sets/match etc. and have transition matrix.

    Notes:
        - This is likely to underestimate the probability of the favorite winning the tiebreak.

    Parameters:
        - p1_serve_win_prob (float): Probability of player 1 winning a point on serve.
        - p2_serve_win_prob (float): Probability of player 2 winning a point on serve.

    Returns:
        - float: Probability of player 1 winning the tiebreak.

    """
    return p1_serve_win_prob / (p1_serve_win_prob + p2_serve_win_prob)


def build_set_transition_matrix(
    p1_service_game_proba, p2_service_game_proba, p1_tiebreak_prob=0.5
):
    """
    Build the transition matrix for a set.

    Parameters:
        - p1_service_game_proba (float): Probability of player 1 winning a service game.
        - p2_service_game_proba (float): Probability of player 2 winning a service game.
        - p1_tiebreak_prob (float): Probability of player 1 winning a tiebreak.

    Returns:
        - np.ndarray: Transition matrix for a set.

    Notes:
        We assume player 1 serves first in the set, the first server does not make a
        difference to winning probability in non-momentum models (does affect totals markets).
    """
    first_server = DEFAULT_FIRST_SERVER
    size = 7**2 + 4
    m = np.zeros((size, size))
    for a in range(8):
        for b in range(8):
            if a == b == 7:
                continue
            if max(a, b) == 7 and min(a, b) < 5:
                continue
            if is_winning_set_score(a, b):
                m[state_to_index_set(a, b), state_to_index_set(a, b)] = 1
            elif a == 6 and b == 6:
                m[state_to_index_set(a, b), state_to_index_set(a + 1, b)] = (
                    p1_tiebreak_prob
                )
                m[state_to_index_set(a, b), state_to_index_set(a, b + 1)] = (
                    1 - p1_tiebreak_prob
                )
            else:
                if is_p1_serving(a, b, first_server):
                    m[state_to_index_set(a, b), state_to_index_set(a + 1, b)] = (
                        p1_service_game_proba
                    )
                    m[state_to_index_set(a, b), state_to_index_set(a, b + 1)] = (
                        1 - p1_service_game_proba
                    )
                else:
                    m[state_to_index_set(a, b), state_to_index_set(a, b + 1)] = (
                        p2_service_game_proba
                    )
                    m[state_to_index_set(a, b), state_to_index_set(a + 1, b)] = (
                        1 - p2_service_game_proba
                    )
    return m


def get_set_final_transition_matrix(
    single_set_transition_matrix: np.ndarray,
) -> np.ndarray:
    """
    Get the final transition matrix for a set.

    Parameters:
        - single_set_transition_matrix (np.ndarray): Transition matrix for a set.

    Returns:
        - np.ndarray: Final transition matrix for a set.

    Notes:
        - Raises the matrix to the power of 13 to get the final transition matrix. (since 13 is the max number of games in a set)
    """
    max_steps = 13
    return np.linalg.matrix_power(single_set_transition_matrix, max_steps)


def final_set_states() -> Generator[tuple[int, int], None, None]:
    """Generate the final set states."""
    for score in range(5):
        yield 6, score
        yield score, 6
    for score in range(5, 7):
        yield 7, score
        yield score, 7


def player_one_winning_set_states() -> Generator[tuple[int, int], None, None]:
    """Generate the states where player 1 wins the set."""
    for state in final_set_states():
        if state[0] > state[1]:
            yield state


def get_player_1_set_winning_probabilities(
    single_set_transition_matrix: np.ndarray,
) -> float:
    """
    Get the probability of player 1 winning the set.

    Parameters:
        - single_set_transition_matrix (np.ndarray): Transition matrix for a set.

    Returns:
        - float: Probability of player 1 winning the set.
    """
    initial_index = state_to_index_set(0, 0)
    final_matrix = get_set_final_transition_matrix(single_set_transition_matrix)
    p1_winning_states = list(player_one_winning_set_states())
    p1_winning_prob = sum(
        final_matrix[initial_index, state_to_index_set(a, b)]
        for a, b in p1_winning_states
    )
    return p1_winning_prob


def get_player_1_set_winning_probability(params: TennisParameters) -> float:
    """
    Get the probability of player 1 winning a set.

    Parameters:
        - params (TennisParameters): The tennis parameters.

    Returns:
        - float: Probability of player 1 winning a set.
    """
    set_transition_matrix = build_set_transition_matrix(
        service_game_winning_prob(params.player_one_point_on_serve_prob),
        service_game_winning_prob(params.player_two_point_on_serve_prob),
    )
    return get_player_1_set_winning_probabilities(set_transition_matrix)


def state_to_index_match(a: int, b: int) -> int:
    """Convert the state (a, b) to an index in the transition matrix.
    Possible states are 0-0,1-0,0-1,1-1,2-0,0-2,2-1,1-2,2-2,3-0,0-3,3-1,1-3,3-2,2-3
    """
    return max(a, b) ** 2 + 2 * min(a, b) + (a < b)


def index_to_state_match(index: int) -> tuple[int, int]:
    """Convert the index in the transition matrix to a state (a, b)."""
    all_states = [(a, b) for a in range(4) for b in range(4)]
    all_states_to_index = {state_to_index_match(a, b): (a, b) for a, b in all_states}
    return all_states_to_index[index]


def is_winning_match_score(a: int, b: int, max_sets_won) -> bool:
    """Check if the match score (a, b) is a winning match score."""
    if b > a:
        return is_winning_match_score(b, a, max_sets_won)
    if a == max_sets_won and b < max_sets_won:
        return True
    return False


def player_one_winning_match_states(max_sets_won):
    """Generate the states where player 1 wins the match."""
    for a in range(max_sets_won + 1):
        for b in range(max_sets_won + 1):
            if a > b:
                if is_winning_match_score(a, b, max_sets_won):
                    yield a, b


def build_match_transition_matrix(p1_set_proba, first_server=1, max_sets_won=3):
    """
    Build the transition matrix for a match.

    Parameters:
        - p1_set_proba (float): Probability of player 1 winning a set.
        - first_server (int): The player who serves first.
        - max_sets_won (int): The maximum number of sets that can be won.

    Returns:
        - np.ndarray: Transition matrix for a match.
    """
    size = max_sets_won * (max_sets_won + 2)
    m = np.zeros((size, size))
    for a in range(max_sets_won + 1):
        for b in range(max_sets_won + 1):
            if a == b == max_sets_won:
                continue
            if is_winning_match_score(a, b, max_sets_won):
                m[state_to_index_match(a, b), state_to_index_match(a, b)] = 1
            else:
                if is_p1_serving(a, b, first_server):
                    m[state_to_index_match(a, b), state_to_index_match(a + 1, b)] = (
                        p1_set_proba
                    )
                    m[state_to_index_match(a, b), state_to_index_match(a, b + 1)] = (
                        1 - p1_set_proba
                    )
                else:
                    m[state_to_index_match(a, b), state_to_index_match(a, b + 1)] = (
                        1 - p1_set_proba
                    )
                    m[state_to_index_match(a, b), state_to_index_match(a + 1, b)] = (
                        p1_set_proba
                    )
    return m


def get_match_final_transition_matrix(
    single_match_transition_matrix: np.ndarray, max_sets: int
) -> np.ndarray:
    """
    Get the final transition matrix for a match.

    Parameters:
        - single_match_transition_matrix (np.ndarray): Transition matrix for a match.
        - max_sets (int): The maximum number of sets that can be won.

    Returns:
        - np.ndarray: Final transition matrix for a match.
    """
    return np.linalg.matrix_power(single_match_transition_matrix, max_sets)


def get_player_1_match_winning_probability_from_transition_matrix(
    single_match_transition_matrix: np.ndarray, max_sets_playable: int
) -> float:
    """
    Get the probability of player 1 winning the match.

    Parameters:
        - single_match_transition_matrix (np.ndarray): Transition matrix for a match.
        - max_sets_playable (int): The maximum number of sets that can be won.

    Returns:
        - float: Probability of player 1 winning the match.
    """
    max_score = (max_sets_playable + 1) // 2
    initial_index = state_to_index_match(0, 0)
    final_matrix = get_match_final_transition_matrix(
        single_match_transition_matrix, max_sets_playable
    )
    p1_winning_states = list(player_one_winning_match_states(max_score))
    p1_winning_prob = sum(
        final_matrix[initial_index, state_to_index_match(a, b)]
        for a, b in p1_winning_states
    )

    return p1_winning_prob


def get_player_1_match_winning_probability(
    params: TennisParameters, max_sets_playable: int
) -> float:
    """
    Get the probability of player 1 winning a match.

    Parameters:
        - params (TennisParameters): The tennis parameters.
        - max_sets_playable (int): The maximum number of sets that can be won.

    Returns:
        - float: Probability of player 1 winning the match.
    """
    max_score = (max_sets_playable + 1) // 2
    set_transition_matrix = build_set_transition_matrix(
        service_game_winning_prob(params.player_one_point_on_serve_prob),
        service_game_winning_prob(params.player_two_point_on_serve_prob),
    )
    match_transition_matrix = build_match_transition_matrix(
        get_player_1_set_winning_probabilities(set_transition_matrix),
        first_server=1,
        max_sets_won=max_score,
    )
    return get_player_1_match_winning_probability_from_transition_matrix(
        match_transition_matrix, max_sets_playable
    )
