import logging
import time

from utils import (
    MATCHES_PER_SCORE_EVALUATION,
    PURPLE,
    _format_duration,
    _load_model,
    _load_model_score_history,
    _model_path,
    _save_model_score_history,
    progress_done,
    progress_line,
)
from run_tournament import _play_match

logger = logging.getLogger(__name__)


def run_score_evaluation(gen_id):
    if gen_id == 0:
        logger.info("[gen 0] score evaluation skipped — no prior generations to compare against")
        return False

    score_history = _load_model_score_history()
    candidates = [row for row in score_history if row[0] != gen_id and row[3] is not None]
    if not candidates:
        logger.warning("[gen %s] score evaluation skipped — no scored opponents available yet", gen_id)
        return False

    # Median opponent selection: sort ascending by score, take the higher of
    # the two middle rows when the count is even (index n // 2 in both cases).
    candidates.sort(key=lambda row: row[3])
    opponent_gen, _, _, opponent_score, _ = candidates[len(candidates) // 2]

    current_model = _load_model(_model_path(gen_id))
    opponent_model = _load_model(_model_path(opponent_gen))

    total_matches = MATCHES_PER_SCORE_EVALUATION
    logger.info(
        "[gen %s] starting score evaluation — %s match(es) vs gen_%s (median score %.3f)",
        gen_id, total_matches, opponent_gen, opponent_score,
    )
    eval_start = time.time()

    wins = 0.0
    for match_index in range(1, total_matches + 1):
        games_current, games_opp = _play_match(current_model, opponent_model)
        if games_current > games_opp:
            wins += 1.0
            outcome = "win"
        elif games_current < games_opp:
            outcome = "loss"
        else:
            wins += 0.5
            outcome = "draw"

        progress_line(
            "score-eval",
            label_color=PURPLE,
            generation=gen_id,
            progress=f"match {match_index}/{total_matches} vs gen_{opponent_gen}",
            result=f"{games_current}-{games_opp} ({outcome}), elapsed {_format_duration(time.time() - eval_start)}",
        )

    progress_done()

    score = wins / total_matches
    for row in score_history:
        if row[0] == gen_id:
            row[3] = score
            break
    _save_model_score_history(score_history)

    logger.info(
        "🎯 [gen %s] score evaluation complete — %s match(es) in %s | score %.3f (win ratio) vs gen_%s",
        gen_id, total_matches, _format_duration(time.time() - eval_start), score, opponent_gen,
    )

    return True
