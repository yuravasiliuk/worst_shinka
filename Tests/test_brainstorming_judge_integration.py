import textwrap

from worst_shinka.judge import Judge, JudgeConfig


def test_brainstorming_proposals_are_compatible_with_judge(tmp_path):
    """
    Integration test:

        Brainstorming
            ↓
        Proposal 1 / Proposal 2
            ↓
        training
            ↓
        model_1.pt / model_2.pt
            ↓
        Judge
            ↓
        evaluation result
    """

    # ---------------------------------------------------------
    # 1. Fake proposals produced by Brainstorming
    # ---------------------------------------------------------

    proposal_1 = textwrap.dedent(
        """
        import random

        def get_epsilon(episode_index, hyperparameters):
            start = hyperparameters["epsilon_start"]
            end = hyperparameters["epsilon_end"]
            decay = max(1, hyperparameters["epsilon_decay_episodes"])

            progress = min(1.0, episode_index / decay)
            return start + (end - start) * progress

        def select_action(model, observation, epsilon, num_actions):
            if random.random() < epsilon:
                return random.randrange(num_actions)

            return int(model.predict(observation).argmax())

        def select_opponent_action(model, observation, num_actions):
            if model is None:
                return random.randrange(num_actions)

            return int(model.predict(observation).argmax())

        def update_model(
            model,
            state,
            action,
            reward,
            next_state,
            done,
            hyperparameters,
        ):
            model.train_step(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                gamma=hyperparameters["gamma"],
            )
        """
    )

    proposal_2 = textwrap.dedent(
        """
        import random

        def get_epsilon(episode_index, hyperparameters):
            start = hyperparameters["epsilon_start"]
            end = hyperparameters["epsilon_end"]
            decay = max(1, hyperparameters["epsilon_decay_episodes"])

            progress = min(1.0, episode_index / decay)
            return start + (end - start) * progress

        def select_action(model, observation, epsilon, num_actions):
            if random.random() < epsilon:
                return random.randrange(num_actions)

            q_values = model.predict(observation)
            return int(q_values.argmax())

        def select_opponent_action(model, observation, num_actions):
            if model is None:
                return random.randrange(num_actions)

            q_values = model.predict(observation)
            return int(q_values.argmax())

        def update_model(
            model,
            state,
            action,
            reward,
            next_state,
            done,
            hyperparameters,
        ):
            model.train_step(
                state,
                action,
                reward,
                next_state,
                done,
                hyperparameters["gamma"],
            )
        """
    )

    # ---------------------------------------------------------
    # 2. Verify that Brainstorming produced actual code
    # ---------------------------------------------------------

    assert isinstance(proposal_1, str)
    assert isinstance(proposal_2, str)

    assert "def get_epsilon" in proposal_1
    assert "def select_action" in proposal_1
    assert "def update_model" in proposal_1

    assert "def get_epsilon" in proposal_2
    assert "def select_action" in proposal_2
    assert "def update_model" in proposal_2

    # ---------------------------------------------------------
    # 3. Fake training stage
    #
    # We don't run real RL training here because this test is
    # intended to verify the integration contract.
    # ---------------------------------------------------------

    model_a = tmp_path / "model_a.pt"
    model_b = tmp_path / "model_b.pt"

    model_a.write_bytes(b"fake-model-a")
    model_b.write_bytes(b"fake-model-b")

    assert model_a.exists()
    assert model_b.exists()

    # ---------------------------------------------------------
    # 4. Fake Judge match evaluation
    #
    # Replace the internal match execution with deterministic
    # results. This allows us to test Judge logic without
    # requiring a long Atari training run.
    # ---------------------------------------------------------

    judge = Judge(
        JudgeConfig(
            win_rate_weight=0.60,
            game_difference_weight=0.25,
            consistency_weight=0.15,
        )
    )

    def fake_evaluate_matches(solution_a, solution_b, games):
        return [
            {
                "match": 0,
                "a_games": 4,
                "b_games": 1,
                "winner": "A",
            },
            {
                "match": 1,
                "a_games": 3,
                "b_games": 2,
                "winner": "A",
            },
            {
                "match": 2,
                "a_games": 4,
                "b_games": 2,
                "winner": "A",
            },
            {
                "match": 3,
                "a_games": 2,
                "b_games": 3,
                "winner": "B",
            },
            {
                "match": 4,
                "a_games": 4,
                "b_games": 1,
                "winner": "A",
            },
        ]

    judge._evaluate_matches = fake_evaluate_matches

    # ---------------------------------------------------------
    # 5. Judge evaluates the two trained candidates
    # ---------------------------------------------------------

    result = judge.evaluate(
        solution_a=str(model_a),
        solution_b=str(model_b),
        games=5,
        id_a="proposal-1",
        id_b="proposal-2",
    )

    # ---------------------------------------------------------
    # 6. Validate Judge result
    # ---------------------------------------------------------

    assert result["winner"] == str(model_a)
    assert result["winner_id"] == "proposal-1"

    assert "metrics" in result
    assert "A" in result["metrics"]
    assert "B" in result["metrics"]

    assert "win_rate" in result["metrics"]["A"]
    assert "game_difference" in result["metrics"]["A"]
    assert "consistency" in result["metrics"]["A"]
    assert "score" in result["metrics"]["A"]

    assert "matches" in result
    assert len(result["matches"]) == 5

    # Proposal 1 won 4/5 matches.
    assert result["metrics"]["A"]["win_rate"] == 0.8

    # Proposal 2 won 1/5.
    assert result["metrics"]["B"]["win_rate"] == 0.2

    # Proposal 1 must have a higher final score.
    assert (
        result["metrics"]["A"]["score"]
        > result["metrics"]["B"]["score"]
    )