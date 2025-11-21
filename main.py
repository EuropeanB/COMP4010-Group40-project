from RunPPOAgent import RunPPOAgent
from RunDQNAgent import RunDQNAgent

if __name__ == "__main__":
    RunPPOAgent(True, save_directory="Models")
    # RunPPOAgent(False, test_model="Models/actual_best_agent", num_tests=100, test_render_mode='human')

    #RunDQNAgent(True, save_directory='Models')
