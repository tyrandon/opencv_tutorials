class Champion:
    id = 0
    cost = 0
    numOwned = 0
    numTaken = 0
    shopImg = None
    inspectImg = None
    traits = []

    def __init__(self, name, cost, numOwned, numTaken, shopImg, inspectImg, traits):
        self.name = name
        self.cost = cost
        self.numOwned = numOwned
        self.numTaken = numTaken
        self.shopImg = shopImg
        self.inspectImg = inspectImg
        self.traits = traits
    
    def setNumOwned(self, numOwned):
        self.numOwned = numOwned

    def setNumTaken(self, numTaken):
        self.numTaken = numTaken