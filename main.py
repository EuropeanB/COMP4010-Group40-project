from RunPPOAgent import RunPPOAgent
from RunDQNAgent import RunDQNAgent

if __name__ == "__main__":
    #RunPPOAgent(True, save_directory="Models")
    #Run ACAgent
    #RunACAgent(True, save_directory="Models")
    #Run PPOAgent
    RunPPOAgent(False, test_model="Models/PPO_BestAgent.pth", num_tests=1000, test_render_mode='human')

    #RunDQNAgent(True, save_directory='Models')

