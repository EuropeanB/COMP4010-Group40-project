from selenium import webdriver
import time

class Visualizer:
    """
    Bridge class that connects the agent to the javascript game environment

    This class uses Selenium WebDriver to control a browser instance running the Dragonsweeper game,
    allowing Python code to send actions and receive game state updates for testing the agent. View the
    ReadMe file for instructions on how to properly run this.
    """

    def __init__(self):
        self.driver = webdriver.Chrome()
        self.driver.get("http://localhost:8000")

    def take_action(self, action):
        # Send the action to the game via postMessage
        self.driver.execute_script(f"""
            window.postMessage({{
                type: 'STEP_ACTION',
                action: {action}
            }}, '*');
        """)

        # Wait for any animations or asynch functions to complete
        time.sleep(0.5)

        # Retrieve the updated game state form the game
        result = self.driver.execute_script("return window.lastGameState;")

        return result
