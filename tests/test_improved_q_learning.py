import unittest

from agent.q_learning import QLearningAgent


class ImprovedQLearningTests(unittest.TestCase):
    def test_food_vector_right_is_detected(self):
        agent = QLearningAgent()
        state = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        dx, dy = agent._food_vector(state)
        self.assertEqual(dx, 1)
        self.assertEqual(dy, 0)

    def test_food_bias_prefers_direction_toward_food(self):
        agent = QLearningAgent()
        state = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
        bias = agent._food_bias(state)
        self.assertGreater(bias[0], bias[1])
        self.assertGreater(bias[0], bias[2])


if __name__ == "__main__":
    unittest.main()
