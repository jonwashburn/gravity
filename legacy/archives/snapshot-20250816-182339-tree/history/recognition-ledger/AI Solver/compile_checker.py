class CompileChecker:
    """Minimal stub that pretends all proofs compile.
    Replace with real Lean compiler invocation when available."""
    
    def check_proof(self, file_path, line_num, proof):
        # In real usage, insert proof into file, run `lean --make` and capture errors.
        # Here we assume success to allow LLM workflow.
        return True, "" 