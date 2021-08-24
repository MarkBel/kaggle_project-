

class VoterClassifier():

    
    def __init__(self, estimators, mode="hard", weight=None, show_info="percent"):
        self.estimators = estimators
        self.mode = mode
        self.weight = weight
        self.show_info = show_info

    
    def predict(self,test):
        if self.mode == "hard":
            pass
        elif self.mode == "soft":
            pass
