from RunPPOAgent import RunPPOAgent
from RunDQNAgent import RunDQNAgent

if __name__ == "__main__":
    #RunPPOAgent(True, save_directory="Models")
    RunPPOAgent(False, test_model="Models/checkpoint_15500.pth", num_tests=1000, test_render_mode='human')

    #RunDQNAgent(True, save_directory='Models')
