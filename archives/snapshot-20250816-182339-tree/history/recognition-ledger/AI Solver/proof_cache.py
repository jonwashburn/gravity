class ProofCache:
    def __init__(self):
        self.cache = {}
    def get_proof(self, declaration):
        return self.cache.get(declaration)
    def store_proof(self, declaration, proof, success):
        if success:
            self.cache[declaration] = proof 