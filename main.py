from RunPPOAgent import RunPPOAgent
from RunDQNAgent import RunDQNAgent

if __name__ == "__main__":
    #RunPPOAgent(True, save_directory="Models")
    RunPPOAgent(False, test_model="Models/up_to_date_best", num_tests=1000, test_render_mode=None)

    #RunDQNAgent(True, save_directory='Models')
