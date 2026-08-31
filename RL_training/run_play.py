
import os
import pygame
import torch
from pettingzoo.atari import tennis_v3
from utils import MAX_CYCLES, _load_model, _model_path

# Settings for our pygame window, because default displays weirdly
RENDER_FPS = 30
CROP_TOP = 4
CROP_LEFT = 8
ZOOM = 4  

# It literally chooses the next action an agent should take based on the current state
# observation - Atari RAM vector (128 bytes), obs_type="ram"
def _select_action(env, agent, observation, model):
    if model is None:
        return env.action_space(agent).sample()
    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        logits = model(obs)
        return int(torch.argmax(logits, dim=-1).item())

def run_play(gen_id_1=None, gen_id_2=None, stop_event=None, model_path=None, opponent_path=None):
    models = {
        "first_0": None,
        "second_0": None,
    }

    try:
        if model_path is not None:
            models["first_0"] = _load_model(model_path)
            if opponent_path is None:
                models["second_0"] = _load_model(model_path)
        elif gen_id_1 is not None:
            path = _model_path(gen_id_1)
            models["first_0"] = _load_model(path)
            if gen_id_2 is None:
                models["second_0"] = _load_model(path)

        if opponent_path is not None:
            models["second_0"] = _load_model(opponent_path)
        elif gen_id_2 is not None:
            path = _model_path(gen_id_2)
            models["second_0"] = _load_model(path)
    except Exception as e:
        print(f"{e}")
        return

    # Creating game visualization
    # Use a much larger cycle limit to allow complete matches (default MAX_CYCLES is for training episodes)
    env = tennis_v3.env(render_mode="rgb_array", obs_type="ram", max_cycles=MAX_CYCLES * 10)
    env.reset()
    ale = env.unwrapped.ale

    screen_h, screen_w = ale.getScreenDims()[1] - CROP_TOP, ale.getScreenDims()[0] - CROP_LEFT
    pygame.display.init()
    screen = pygame.display.set_mode((screen_w * ZOOM, screen_h * ZOOM))
    clock = pygame.time.Clock()

    # Main game loop
    agents_list = env.agents.copy()  # Keep track of all agents to ensure both act each frame
    for agent in env.agent_iter():
        if stop_event is not None and stop_event.is_set():
            break
        if any(
            event.type == pygame.QUIT
            or (event.type == pygame.KEYDOWN and event.key == pygame.K_q)
            for event in pygame.event.get()
        ):
            break

        # gen informations for agent to make action based on these
        observation, _reward, termination, truncation, _info = env.last()

        # Check if current agent is the second one - both agents already did an action for current frame
        is_last_in_round = agent == agents_list[-1]

        # Check if end of the game, if not, select an action for an agent
        if termination or truncation:
            action = None
        else:
            action = _select_action(env, agent, observation, models[agent])
        env.step(action)

        # If it's second agent, render game frame
        if is_last_in_round:
            frame = ale.getScreenRGB()[CROP_TOP:, CROP_LEFT:]
            surf = pygame.image.frombuffer(frame.tobytes(), frame.shape[:2][::-1], "RGB")
            surf = pygame.transform.scale(surf, (screen_w * ZOOM, screen_h * ZOOM))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            clock.tick(RENDER_FPS)

    # End of the game
    env.close()
    pygame.quit()

if __name__ == "__main__":
    run_play()
