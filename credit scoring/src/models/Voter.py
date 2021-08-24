from typing import List, Any


class Voter():

    def __init__(self, estimators: List[str, Any], mode: str = "hard", task: str = "classification"):
        self.estimators = estimators
        self.mode = mode
        self.task = task


    def weighted_voter(self, weight: List[float] = None,):
        pass

    def hard_voter(self):
        max of list
        pass

    def 