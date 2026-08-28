from worst_shinka.cli import integrations
from worst_shinka.judge import Judge


def main():
    proposal_1 = """
def get_epsilon(episode_index, hyperparameters):
    return hyperparameters["epsilon_end"]


def select_action(model, observation, epsilon, num_actions):
    return 0


def select_opponent_action(model, observation, num_actions):
    return 0


def update_model(model, state, action, reward, next_state, done, hyperparameters):
    model.train_step(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        gamma=hyperparameters["gamma"],
    )
"""

    proposal_2 = """
def get_epsilon(episode_index, hyperparameters):
    return hyperparameters["epsilon_end"]


def select_action(model, observation, epsilon, num_actions):
    return num_actions - 1


def select_opponent_action(model, observation, num_actions):
    return num_actions - 1


def update_model(model, state, action, reward, next_state, done, hyperparameters):
    model.train_step(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        gamma=hyperparameters["gamma"],
    )
"""

    # Пока проверяем только реальную integration-функцию,
    # которая существует в brainstorming branch.
    candidates = integrations.train_and_evaluate(
        proposals=[
            {
                "id": "proposal-A",
                "code": proposal_1,
            },
            {
                "id": "proposal-B",
                "code": proposal_2,
            },
        ],
        workers=1,
    )

    print("\n========== TRAINING RESULT ==========")
    print(candidates)

    # Judge получает уже подготовленные candidates.
    judge = Judge()

    print("\n========== JUDGE ==========")

    result = integrations.judge_candidates(
        candidates=candidates,
    )

    print("\n========== WINNER ==========")
    print(result)


if __name__ == "__main__":
    main()