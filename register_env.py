from gymnasium.envs.registration import register

def register_dragonsweeper():
    # only register if not already registered to avoid duplicate registration errors
    try:
        register(
            id="Dragonsweeper-v0",
            entry_point="Environment:DragonSweeperEnv",
        )
    except Exception:
        # ignore if already registered
        pass